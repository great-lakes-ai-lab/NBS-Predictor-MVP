from src.data_loading import clean_file_name


def test_clean_file_name():
    output1 = clean_file_name("CFS_APCP_Basin_Avgs.csv")
    assert output1 == 'apcp_forecasts'
