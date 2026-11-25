"""补传演练核心逻辑：切块、异或帧规划（迭代加深搜索）、轮次执行。

约定：
- 图被切成 4 KiB 块，至多 8 块，末块补零，重组后按原始长度截断。
- 一帧要么是单块 ("single", c, None)，要么是两块异或 ("xor", a, b)。
- 终端实收异或帧时，恰缺两块之一且本轮解码额度未用完，才解回缺块（计 1 次解码）。
- 规划按"本轮无丢帧"推算：所有终端都收到所有帧。

规划器说明：
- 只要存在一串帧能在帧数与各站解码额度内让各站收齐，就一定找到并返回；
  确实无解才返回 None。已收齐时返回 []。
- 搜索为迭代加深的深度优先（IDA* 式）：单块帧彼此可交换、且任意可行序列
  都可重排为"单块在前、异或在后"而不增加任何终端的解码次数，故单块按块号
  递增规范化、异或阶段不再出现单块——在不丢解的前提下大幅压缩搜索空间。
- 剪枝下界（均可采纳）：各站缺块数的最大值、缺块总数/台数、缺失并集块数
  减去剩余解码额度总和、缺失并集块数的一半（一帧至多覆盖两个缺失并集块）。
- 转置表去重，条目占用全局搜索预算：同一服务所有并发规划留存的搜索状态
  合计不超过 SEARCH_NODE_CAP（30 万）；预算耗尽时仅退化为不缓存，搜索仍完备。
"""
import threading
from itertools import combinations

CHUNK_SIZE = 4096
MAX_CHUNKS = 8
MAX_FRAME_LIMIT = 16      # 单轮帧数上限的硬顶，防止搜索失控
SEARCH_NODE_CAP = 300_000  # 同一服务同时留存的搜索状态总数上限

# 全局搜索状态预算：所有并发规划的转置表条目合计不超过 SEARCH_NODE_CAP。
# 用信号量实现，规划结束即释放；获取不到时该次规划仅少缓存，不影响完备性。
_search_budget = threading.BoundedSemaphore(SEARCH_NODE_CAP)


class SearchAborted(Exception):
    """规划被外部中止（例如演练状态已变化，结果注定被丢弃）。"""


def split_chunks(data: bytes):
    """切成 4 KiB 块（末块补零）。返回 (chunks, 原始长度)。"""
    if not data:
        raise ValueError("空文件，无法切块")
    n = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE
    if n > MAX_CHUNKS:
        raise ValueError(
            f"图共 {n} 块，超过 {MAX_CHUNKS} 块上限（需 ≤ {CHUNK_SIZE * MAX_CHUNKS} 字节）"
        )
    chunks = [
        data[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE].ljust(CHUNK_SIZE, b"\0")
        for i in range(n)
    ]
    return chunks, len(data)


def reassemble(chunks, size: int) -> bytes:
    """按原始长度截断重组。"""
    return b"".join(chunks)[:size]


def _apply_frame(masks, decs, quotas, frame):
    """全体终端收一帧后的联合状态与受益台数；无人受益则 benefit 为 0。"""
    kind, a, b = frame
    new_masks = list(masks)
    new_decs = list(decs)
    benefit = 0
    if kind == "single":
        bit = 1 << a
        for i, m in enumerate(masks):
            if not m & bit:
                new_masks[i] = m | bit
                benefit += 1
    else:
        for i, m in enumerate(masks):
            ha = (m >> a) & 1
            hb = (m >> b) & 1
            if ha != hb and decs[i] < quotas[i]:
                miss = b if ha else a
                new_masks[i] = m | (1 << miss)
                new_decs[i] = decs[i] + 1
                benefit += 1
    return tuple(new_masks), tuple(new_decs), benefit


def plan_round(owned_masks, quotas, frame_limit, n_chunks, should_abort=None):
    """求一套帧序列：帧数 ≤ frame_limit，各终端解码 ≤ quotas，且按无丢帧
    推算所有终端收齐。返回帧列表（最短者）；无解返回 None；已收齐返回 []。

    should_abort：可选回调，返回 True 时抛出 SearchAborted 提前终止
    （调用方据此丢弃过期规划，避免空耗全局搜索预算）。
    """
    full = (1 << n_chunks) - 1
    masks0 = tuple(owned_masks)
    quotas = tuple(int(q) for q in quotas)
    if all(m == full for m in masks0):
        return []
    frame_limit = max(1, min(int(frame_limit), MAX_FRAME_LIMIT))
    nterm = len(masks0)

    missing_union0 = 0
    for m in masks0:
        missing_union0 |= full ^ m
    # 候选单块：缺失并集里的块（按块号递增，供规范化枚举）
    single_chunks = [c for c in range(n_chunks) if (missing_union0 >> c) & 1]
    # 候选异或对：至少含一块缺失并集块（两块都不缺的对永远无人能受益）
    xor_pairs = [(a, b) for a, b in combinations(range(n_chunks), 2)
                 if ((missing_union0 >> a) & 1) or ((missing_union0 >> b) & 1)]
    cand_frames = ([("single", c, None) for c in single_chunks]
                   + [("xor", a, b) for a, b in xor_pairs])

    def lower_bound(masks, decs):
        """还需帧数的可采纳下界。"""
        max_miss = 0
        total_miss = 0
        union = 0
        for m in masks:
            miss = full ^ m
            union |= miss
            cnt = miss.bit_count()
            total_miss += cnt
            if cnt > max_miss:
                max_miss = cnt
        u = union.bit_count()
        quota_left = 0
        for q, d in zip(quotas, decs):
            quota_left += max(0, q - d)
        lb = max_miss                                   # 每帧至多给某站 1 块
        lb = max(lb, -(-total_miss // nterm))           # 每帧至多 nterm 块次投递
        lb = max(lb, u - quota_left)                    # 异或帧必耗解码额度
        lb = max(lb, -(-u // 2))                        # 每帧至多覆盖 2 个并集块
        return lb

    # 转置表：本搜索留存的状态 → 以该剩余帧数访问过。受全局预算约束。
    tt = {}
    acquired = 0

    def tt_seen(key, f_rem):
        return tt.get(key, -1) >= f_rem

    def tt_store(key, f_rem):
        nonlocal acquired
        old = tt.get(key)
        if old is not None:
            if f_rem > old:
                tt[key] = f_rem
            return
        if not _search_budget.acquire(blocking=False):
            return  # 全局预算耗尽：不缓存继续搜，完备性不受影响
        acquired += 1
        tt[key] = f_rem

    sys_limit = min(frame_limit, len(single_chunks))  # 纯单块必可行，无需更深
    abort_check = should_abort if should_abort is not None else (lambda: False)

    def dfs(masks, decs, f_rem, singles_from, path):
        """singles_from：单块规范化游标。≥0 表示仍在单块阶段（下一单块块号
        须大于它）；为 None 表示已进入异或阶段（不再发单块）。"""
        if abort_check():
            raise SearchAborted()
        if lower_bound(masks, decs) > f_rem:
            return None
        key = (masks, decs, singles_from)
        if tt_seen(key, f_rem):
            return None
        # 生成候选：本帧无人受益的不可能出现在最短解里，直接跳过
        moves = []
        if singles_from is not None:
            for c in single_chunks:
                if c <= singles_from:
                    continue
                nm, nd, benefit = _apply_frame(masks, decs, quotas,
                                               ("single", c, None))
                if benefit:
                    moves.append((-benefit, ("single", c, None), nm, nd, c))
        for a, b in xor_pairs:
            nm, nd, benefit = _apply_frame(masks, decs, quotas, ("xor", a, b))
            if benefit:
                moves.append((-benefit, ("xor", a, b), nm, nd, None))
        moves.sort(key=lambda mv: mv[0])  # 受益大的帧优先，保持同序稳定
        for _neg_benefit, fr, nm, nd, next_single in moves:
            if all(m == full for m in nm):
                return path + [fr]
            if f_rem > 1:
                got = dfs(nm, nd, f_rem - 1, next_single, path + [fr])
                if got is not None:
                    return got
        tt_store(key, f_rem)
        return None

    try:
        for limit in range(lower_bound(masks0, (0,) * nterm), sys_limit + 1):
            got = dfs(masks0, (0,) * nterm, limit, -1, [])
            if got is not None:
                return got
        return None
    finally:
        if acquired:
            _search_budget.release(acquired)


def apply_round(owned_masks, quotas, frames, fault=None):
    """实发一轮。fault=(终端序号, 帧号1基) 表示该终端漏收该帧。
    返回每终端结果：[{mask, decodes, decoded, received, missed_frame}]。"""
    results = []
    for i, m0 in enumerate(owned_masks):
        m = m0
        decs = 0
        decoded = []    # [(帧号, 解回的块)]
        received = []   # 实收帧号
        missed = fault[1] if fault and fault[0] == i else None
        for no, (kind, a, b) in enumerate(frames, 1):
            if no == missed:
                continue
            received.append(no)
            if kind == "single":
                m |= 1 << a
            else:
                ha = (m >> a) & 1
                hb = (m >> b) & 1
                if ha != hb and decs < quotas[i]:
                    miss = b if ha else a
                    m |= 1 << miss
                    decs += 1
                    decoded.append((no, miss))
        results.append({
            "mask": m,
            "decodes": decs,
            "decoded": decoded,
            "received": received,
            "missed_frame": missed,
        })
    return results
