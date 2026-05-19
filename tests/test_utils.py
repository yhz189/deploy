from utils import to_coarse, CLASSES


def test_to_coarse_dairy():
    # Butter / Cheese / Cooking cream 都应归入「乳品」
    butter_id = CLASSES.index('Butter')
    cheese_id = CLASSES.index('Cheese')
    assert to_coarse(butter_id) == '乳品'
    assert to_coarse(cheese_id) == '乳品'


def test_to_coarse_produce():
    assert to_coarse(CLASSES.index('Apple')) == '蔬果'
    assert to_coarse(CLASSES.index('Tomato')) == '蔬果'


def test_to_coarse_covers_all_classes():
    # 17 个细类每个都要有粗类映射
    for cid in range(len(CLASSES)):
        assert to_coarse(cid) in ('蔬果', '生鲜', '乳品', '包装食品')
