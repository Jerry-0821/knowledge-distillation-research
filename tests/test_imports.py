def test_package_and_subpackages_can_be_imported() -> None:
    import kd_research
    import kd_research.data
    import kd_research.evaluation
    import kd_research.models
    import kd_research.training
    import kd_research.utils

    assert kd_research.__version__ == "0.1.0"
