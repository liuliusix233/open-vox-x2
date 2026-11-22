"""地点记忆冒烟：围栏匹配 + near_here 检索 + 访客硬拦截 + 上下文不含坐标。"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vox_v2.brain import GUEST_BLOCK_MSG, VoxBrain
from vox_v2.config import VoxConfig
from vox_v2.geo import haversine_m, wgs84_to_gcj02
from vox_v2.memory import MemoryStore
from vox_v2.vision import VisionStore


async def run() -> None:
    # 1) 坐标纠偏 + 距离
    glat, glon = wgs84_to_gcj02(31.5500, 120.3000)
    assert abs(glat - 31.55) < 0.01 and abs(glon - 120.30) < 0.01
    assert 1000 < haversine_m(31.55, 120.30, 31.56, 120.30) < 1200

    with tempfile.TemporaryDirectory() as tmp:
        cfg = VoxConfig(db_path=os.path.join(tmp, 't.db'), frame_dir=tmp)
        mem = MemoryStore(cfg.db_path)
        vis = VisionStore(tmp, ttl_s=60)

        # 2) 围栏匹配
        mem.save_place('c1', '公司', glat, glon, 100)
        assert mem.match_place('c1', glat + 0.0003, glon) is not None, '100m 内应命中'
        assert mem.match_place('c1', glat + 0.01, glon) is None, '1.1km 外不应命中'
        hit = mem.match_place('c1', glat, glon)
        assert hit and hit['name'] == '公司'

        # 3) near_here 过滤：写两条视觉单元，一近一远
        now = time.time()
        mem._conn.execute(
            "INSERT INTO memory_units(client_id,kind,start_ts,end_ts,search_text,caption_clean,lat,lon,acc)"
            " VALUES ('c1','vision_episode',?,?,'附近有咖啡店','附近有咖啡店',?,?,10)",
            (now - 60, now - 50, glat, glon),
        )
        mem._conn.execute(
            "INSERT INTO memory_units(client_id,kind,start_ts,end_ts,search_text,caption_clean,lat,lon,acc)"
            " VALUES ('c1','vision_episode',?,?,'远处有加油站','远处有加油站',?,?,10)",
            (now - 120, now - 110, glat + 0.05, glon),
        )
        mem._conn.commit()
        near = mem.search_memory('c1', query='', start_ts=now - 300, end_ts=now,
                                 near_lat=glat, near_lon=glon, near_radius_m=150)
        texts = json.dumps(near, ensure_ascii=False)
        assert '附近有咖啡店' in texts, '150m 内记录应命中'
        assert '远处有加油站' not in texts, '远场记录不该出现'

        # 4) 工具 + 上下文
        captured = {}
        async def mock_llm(messages, tools=None, tool_choice=None):
            captured['messages'] = [dict(m) for m in messages]
            return {'role': 'assistant', 'content': '好'}

        b = VoxBrain(cfg, mem, vis, llm_caller=mock_llm)
        b.location_fn = lambda: {'lat': glat, 'lon': glon, 'acc': 10, 'place': '公司'}

        await b.run_turn('c1', '我在哪', epoch_id=1)
        sys_blob = json.dumps(captured['messages'], ensure_ascii=False)
        assert '[PLACE] 公司' in sys_blob, '上下文缺少 [PLACE] 行'
        assert ('%.5f' % glat) not in sys_blob and ('%.5f' % glon) not in sys_blob, '上下文泄露了原始坐标'

        saved = await b._exec_tool('c1', 'save_place', {'name': '健身房'}, [], guest=False)
        assert json.loads(saved).get('ok') is True, 'save_place 应成功'
        where = json.loads(await b._exec_tool('c1', 'where_am_i', {}, [], guest=False))
        assert where.get('place') in ('公司', '健身房'), where
        assert '公司' in json.dumps(where, ensure_ascii=False), '应给出到公司距离'

        # 5) 访客硬拦截
        GUEST = True
        for name, args in (('save_place', {'name': 'x'}), ('where_am_i', {}),
                           ('search_memory', {'near_here': True})):
            out = await b._exec_tool('c1', name, args, [], guest=True)
            assert GUEST_BLOCK_MSG in out, '%s 未被访客拦截: %s' % (name, out)
        # 访客上下文不出现 [PLACE]
        captured.clear()
        await b.run_turn('c1', '这是哪里', guest_mode=True, guest_since_ts=time.time())
        assert '[PLACE]' not in json.dumps(captured['messages'], ensure_ascii=False), '访客上下文泄露了地点'
        mem.close()

    print('ALL PLACES / NEARBY / GUEST-FENCE SMOKE TESTS PASSED!')


if __name__ == '__main__':
    asyncio.run(run())