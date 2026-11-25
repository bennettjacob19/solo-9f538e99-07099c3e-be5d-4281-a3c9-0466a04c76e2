"""补传演练测试：切块、规划器（紧额度）、漏帧补传、Flask 全流程。"""
import hashlib
import io
import threading

import pytest

import app as web
import planner
from planner import (CHUNK_SIZE, MAX_CHUNKS, apply_round, plan_round,
                     reassemble, split_chunks)


# ---------- 切块 ----------

def test_split_and_reassemble_roundtrip():
    data = bytes(range(256)) * 35 + b"tail"  # 8964 字节 → 3 块
    chunks, size = split_chunks(data)
    assert len(chunks) == 3
    assert all(len(c) == CHUNK_SIZE for c in chunks)
    assert reassemble(chunks, size) == data


def test_split_limits():
    chunks, _ = split_chunks(b"x" * (CHUNK_SIZE * MAX_CHUNKS))
    assert len(chunks) == MAX_CHUNKS
    with pytest.raises(ValueError):
        split_chunks(b"x" * (CHUNK_SIZE * MAX_CHUNKS + 1))
    with pytest.raises(ValueError):
        split_chunks(b"")


# ---------- 规划器 ----------

def test_plan_xor_saves_frame_when_quota_allows():
    # T0 缺块3，T1 缺块2：一帧异或即可双双解回，但各需 1 次解码额度
    owned = [0b011, 0b101]
    plan = plan_round(owned, [1, 1], frame_limit=1, n_chunks=3)
    assert plan == [("xor", 1, 2)]


def test_plan_tight_quota_no_solution_then_exact():
    owned = [0b011, 0b101]
    # 解码额度为 0、只许 1 帧：无解
    assert plan_round(owned, [0, 0], frame_limit=1, n_chunks=3) is None
    # 额度为 0、放宽到 2 帧：退化为两帧单块
    plan = plan_round(owned, [0, 0], frame_limit=2, n_chunks=3)
    assert plan is not None and len(plan) == 2
    assert all(fr[0] == "single" for fr in plan)


def test_plan_tight_frame_limit():
    # 三台各缺两块，最少 3 帧（三块单播）
    owned = [0b001, 0b010, 0b100]
    assert plan_round(owned, [8, 8, 8], frame_limit=2, n_chunks=3) is None
    plan = plan_round(owned, [8, 8, 8], frame_limit=3, n_chunks=3)
    assert plan is not None and len(plan) == 3


def test_plan_already_complete():
    assert plan_round([0b111, 0b111], [0, 0], frame_limit=4, n_chunks=3) == []


# ---------- 8 块 × 4 终端：不再误报无方案 ----------

def _masks_missing(missing_sets, n=8):
    """按各站缺块集合（0 基）构造已收位掩码列表。"""
    full = (1 << n) - 1
    out = []
    for miss in missing_sets:
        m = full
        for c in miss:
            m &= ~(1 << c)
        out.append(m)
    return out


def test_plan_eight_chunks_four_terminals():
    # 缺失并集为全部 8 块，帧限 8：纯单块必可行，绝不可报无方案
    masks = _masks_missing([{0, 1, 2, 3, 4}, {2, 3, 4, 5, 6},
                            {0, 4, 5, 6, 7}, {1, 3, 5, 7}])
    quotas = [2, 2, 2, 2]
    plan = plan_round(masks, quotas, frame_limit=8, n_chunks=8)
    assert plan is not None and 0 < len(plan) <= 8
    res = apply_round(masks, quotas, plan)
    assert all(r["mask"] == 0xFF for r in res)
    assert all(r["decodes"] <= q for r, q in zip(res, quotas))


def test_plan_eight_chunks_xor_compression():
    # 缺失并集 8 块 > 帧限 7：必须靠异或压缩；额度够则有解
    masks = _masks_missing([{0, 4}, {1, 5}, {2, 6}, {3, 7}])
    quotas = [1, 1, 1, 1]
    plan = plan_round(masks, quotas, frame_limit=7, n_chunks=8)
    assert plan is not None and len(plan) <= 7
    res = apply_round(masks, quotas, plan)
    assert all(r["mask"] == 0xFF for r in res)


def test_plan_truly_unsolvable_still_none():
    # 缺失并集 8 块 > 帧限 7，且零解码额度（异或无效）→ 确实无解
    masks = _masks_missing([{0, 1}, {2, 3}, {4, 5}, {6, 7}])
    assert plan_round(masks, [0, 0, 0, 0], frame_limit=7, n_chunks=8) is None
    # 放宽到 8 帧：纯单块即可行
    plan = plan_round(masks, [0, 0, 0, 0], frame_limit=8, n_chunks=8)
    assert plan is not None and len(plan) == 8
    assert all(fr[0] == "single" for fr in plan)


def test_plan_should_abort_raises():
    with pytest.raises(planner.SearchAborted):
        plan_round([0b011, 0b101], [1, 1], 1, 3, should_abort=lambda: True)


def test_search_budget_released_after_plan():
    # 规划结束后全局搜索状态预算须全部归还（上限 30 万）
    assert planner._search_budget._value == planner.SEARCH_NODE_CAP == 300_000
    masks = _masks_missing([{0, 1, 2}, {3, 4, 5}, {6, 7}, {0, 7}])
    plan_round(masks, [2, 2, 2, 2], frame_limit=8, n_chunks=8)
    assert planner._search_budget._value == planner.SEARCH_NODE_CAP


# ---------- 轮次执行 ----------

def test_apply_round_fault_injection():
    frames = [("xor", 1, 2)]
    res = apply_round([0b011, 0b101], [1, 1], frames, fault=(0, 1))
    assert res[0]["mask"] == 0b011 and res[0]["decodes"] == 0  # 漏收，原地踏步
    assert res[1]["mask"] == 0b111 and res[1]["decodes"] == 1  # 解回块2
    assert res[1]["decoded"] == [(1, 1)]


def test_apply_round_quota_exhausted_no_decode():
    frames = [("xor", 1, 2)]
    res = apply_round([0b011, 0b101], [0, 0], frames)
    assert res[0]["mask"] == 0b011 and res[1]["mask"] == 0b101


# ---------- Flask 全流程 ----------

ORIGINAL = (b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 34 + b"demo")  # 3 块


@pytest.fixture
def client():
    with web.LOCK:
        web.STATE.clear()
        web.STATE.update(web._fresh_state())
    web.app.config.update(TESTING=True)
    return web.app.test_client()


def _upload(client, data=ORIGINAL, name="new.png"):
    return client.post("/upload", data={"image": (io.BytesIO(data), name)},
                       content_type="multipart/form-data", follow_redirects=True)


def _setup_two_terminals(client):
    # T1 首播收到 #1#2（缺#3），T2 收到 #1#3（缺#2）
    return client.post("/terminals", data={
        "enable_0": "on", "name_0": "T1", "recv_0": ["1", "2"],
        "enable_1": "on", "name_1": "T2", "recv_1": ["1", "3"],
    }, follow_redirects=True)


def test_full_flow_xor_round_completes_all(client):
    assert b"3" in _upload(client).data  # 切为 3 块
    _setup_two_terminals(client)
    r = client.post("/round", data={"frame_limit": "1", "quota_0": "1", "quota_1": "1"},
                    follow_redirects=True)
    assert "已发送 1 帧".encode() in r.data
    with web.LOCK:
        terms = list(web.STATE["terminals"])
        rounds = list(web.STATE["rounds"])
    assert all(t["completed"] for t in terms)
    assert rounds[0]["frames"] == ["异或 #2⊕#3"]
    # 两台都通过校验并替换旧图，可下载且与新图一致
    for tid in (0, 1):
        dl = client.get(f"/terminal/{tid}/download")
        assert dl.status_code == 200
        assert dl.data == ORIGINAL
        assert hashlib.sha256(dl.data).hexdigest() == web.STATE["image"]["sha256"]


def test_no_solution_keeps_state(client):
    _upload(client)
    _setup_two_terminals(client)
    r = client.post("/round", data={"frame_limit": "1", "quota_0": "0", "quota_1": "0"},
                    follow_redirects=True)
    assert "无方案".encode() in r.data
    with web.LOCK:
        assert web.STATE["rounds"] == []
        assert web.STATE["terminals"][0]["mask"] == 0b011
        assert web.STATE["terminals"][1]["mask"] == 0b101


def test_missed_frame_then_next_round_repair(client):
    _upload(client)
    _setup_two_terminals(client)
    # 第 1 轮：T1 漏收唯一的异或帧
    r = client.post("/round", data={
        "frame_limit": "1", "quota_0": "1", "quota_1": "1",
        "fault_terminal": "0", "fault_frame": "1",
    }, follow_redirects=True)
    assert "漏收第1帧".encode() in r.data
    with web.LOCK:
        t1, t2 = web.STATE["terminals"]
        assert not t1["completed"] and t1["mask"] == 0b011  # 仍缺 #3
        assert t2["completed"]
    dl = client.get("/terminal/0/download")
    assert dl.data != ORIGINAL  # 旧图未被替换
    # 第 2 轮：按实收状态补传，单块 #3 即可，无需解码额度
    r = client.post("/round", data={"frame_limit": "2", "quota_0": "0", "quota_1": "0"},
                    follow_redirects=True)
    assert "已发送 1 帧".encode() in r.data
    with web.LOCK:
        assert web.STATE["rounds"][1]["frames"] == ["单块 #3"]
        assert web.STATE["terminals"][0]["completed"]
    assert client.get("/terminal/0/download").data == ORIGINAL


def test_upload_oversize_rejected(client):
    big = b"y" * (CHUNK_SIZE * MAX_CHUNKS + 1)
    r = _upload(client, data=big, name="big.bin")
    assert "超过".encode() in r.data or "上限".encode() in r.data
    with web.LOCK:
        assert web.STATE["image"] is None


# ---------- 8 块 × 4 终端全流程 ----------

def test_web_eight_chunks_four_terminals_flow(client):
    data = bytes(range(256)) * 128  # 32768 字节 → 8 块
    r = _upload(client, data=data, name="map8.bin")
    assert "8 块".encode() in r.data
    r = client.post("/terminals", data={
        "enable_0": "on", "name_0": "A", "recv_0": ["1", "2", "3", "4"],
        "enable_1": "on", "name_1": "B", "recv_1": ["1", "2", "5", "6"],
        "enable_2": "on", "name_2": "C", "recv_2": ["3", "4", "7", "8"],
        "enable_3": "on", "name_3": "D", "recv_3": ["5", "6", "7", "8"],
    }, follow_redirects=True)
    assert "已登记 4 台".encode() in r.data
    # 每台缺 4 块、缺失并集为 8 块：帧限 8、各站额度 2，必须有方案并实发
    r = client.post("/round", data={
        "frame_limit": "8",
        "quota_0": "2", "quota_1": "2", "quota_2": "2", "quota_3": "2",
    }, follow_redirects=True)
    assert "无方案：".encode() not in r.data  # 闪出的是「无方案」提示而非静态文案
    assert "已发送".encode() in r.data
    with web.LOCK:
        assert len(web.STATE["rounds"]) == 1
        assert len(web.STATE["rounds"][0]["frames"]) <= 8
        assert all(t["completed"] for t in web.STATE["terminals"])
    for tid in range(4):
        dl = client.get(f"/terminal/{tid}/download")
        assert dl.status_code == 200
        assert dl.data == data
        assert hashlib.sha256(dl.data).hexdigest() == web.STATE["image"]["sha256"]


def test_web_truly_unsolvable_keeps_state(client):
    data = bytes(range(256)) * 128  # 8 块
    _upload(client, data=data, name="map8.bin")
    client.post("/terminals", data={
        "enable_0": "on", "name_0": "A", "recv_0": ["1", "2"],
        "enable_1": "on", "name_1": "B", "recv_1": ["3", "4"],
        "enable_2": "on", "name_2": "C", "recv_2": ["5", "6"],
        "enable_3": "on", "name_3": "D", "recv_3": ["7", "8"],
    }, follow_redirects=True)
    # 缺失并集 8 块 > 帧限 7，零额度 → 确实无解
    r = client.post("/round", data={
        "frame_limit": "7",
        "quota_0": "0", "quota_1": "0", "quota_2": "0", "quota_3": "0",
    }, follow_redirects=True)
    assert "无方案".encode() in r.data
    with web.LOCK:
        assert web.STATE["rounds"] == []
        masks = [t["mask"] for t in web.STATE["terminals"]]
    assert masks == [0b00000011, 0b00001100, 0b00110000, 0b11000000]


# ---------- 规划期间不阻塞、过期结果丢弃 ----------

def test_planning_releases_lock_and_stale_result_discarded(client, monkeypatch):
    _upload(client)
    _setup_two_terminals(client)
    started = threading.Event()
    release = threading.Event()
    real_plan = web.plan_round
    calls = {"n": 0}

    def slow_plan(masks, quotas, frame_limit, n, should_abort=None):
        calls["n"] += 1
        if calls["n"] == 1:
            started.set()
            assert release.wait(5)
        # 不转发 should_abort：模拟"正常算完"的旧规划，走纪元校验丢弃
        return real_plan(masks, quotas, frame_limit, n)

    monkeypatch.setattr(web, "plan_round", slow_plan)
    other = web.app.test_client()  # 另一会话，模拟并发操作者
    holder = {}

    def first_round():
        holder["resp"] = other.post("/round", data={
            "frame_limit": "1", "quota_0": "1", "quota_1": "1",
        }, follow_redirects=True)

    th = threading.Thread(target=first_round)
    th.start()
    assert started.wait(5)
    # 规划期间无需等计算结束：可下载终端当前图
    assert client.get("/terminal/0/download").status_code == 200
    # 可上传新图
    new_data = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 34 + b"v2"
    _upload(client, data=new_data)
    # 可登记终端并开始新一轮补传（本次规划不被阻塞，直接落地）
    _setup_two_terminals(client)
    r2 = client.post("/round", data={
        "frame_limit": "1", "quota_0": "1", "quota_1": "1",
    }, follow_redirects=True)
    assert "已发送 1 帧".encode() in r2.data
    release.set()
    th.join(5)
    assert not th.is_alive()
    # 旧规划结果已过期：不得发帧，不得改轮次记录、终端进度
    assert "已丢弃".encode() in holder["resp"].data
    with web.LOCK:
        assert len(web.STATE["rounds"]) == 1  # 只有新一轮的记录
        assert web.STATE["rounds"][0]["no"] == 1
        terms = list(web.STATE["terminals"])
    assert all(t["completed"] for t in terms)
    for tid in (0, 1):
        assert client.get(f"/terminal/{tid}/download").data == new_data


def test_concurrent_rounds_only_one_applies(client, monkeypatch):
    # 两轮基于同一状态的规划并发：先落地者生效，后到者按过期丢弃
    _upload(client)
    _setup_two_terminals(client)
    barrier = threading.Barrier(2, timeout=5)
    real_plan = web.plan_round

    def gated_plan(masks, quotas, frame_limit, n, should_abort=None):
        barrier.wait()  # 等两个请求都进入规划，确保二者快照同一纪元
        return real_plan(masks, quotas, frame_limit, n)

    monkeypatch.setattr(web, "plan_round", gated_plan)
    c1, c2 = web.app.test_client(), web.app.test_client()
    holder = {}

    def do_round(cli, key):
        holder[key] = cli.post("/round", data={
            "frame_limit": "1", "quota_0": "1", "quota_1": "1",
        }, follow_redirects=True)

    t1 = threading.Thread(target=do_round, args=(c1, "a"))
    t2 = threading.Thread(target=do_round, args=(c2, "b"))
    t1.start()
    t2.start()
    t1.join(5)
    t2.join(5)
    assert not t1.is_alive() and not t2.is_alive()
    bodies = holder["a"].data + holder["b"].data
    assert "已发送 1 帧".encode() in bodies
    assert "已丢弃".encode() in bodies
    with web.LOCK:
        assert len(web.STATE["rounds"]) == 1
        assert all(t["completed"] for t in web.STATE["terminals"])
