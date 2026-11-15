from continual_manipulation.task_banks import backend_tasks, bank_hashes


def test_metaworld_reach_bank_hash_regression() -> None:
    hashes = bank_hashes(backend_tasks("reach-v3", 17))
    assert hashes[:3] == [
        "6f4437c1e09693dc93858da52b44abe17c799681d20f1990fa25b18918527300",
        "7172f61a01ef804fbe71a517786686d4f676f4023d502d3d69b41df1675dec48",
        "086a6079431bc27097178219191e002064b519838d6a392f7140416c187821f3",
    ]
