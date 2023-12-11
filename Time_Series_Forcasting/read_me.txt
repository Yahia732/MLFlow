The lags folder contains the number of lags each dataset needs make sure then you send at least that number.
if there's nan values make sure there is at least non nan values that is equal to the number of lags needed.
request body format example:
{
    "dataset_id": 9,
"values": [
        {
            "time":"7/1/2021  12:00:00 AM", "value": -0.925346520644351
        }
    ]
}

