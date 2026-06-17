import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from interface.main import pred

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/predict")
def predict(
    AccountAge: int,
    MonthlyCharges: float,
    TotalCharges: float,
    SubscriptionType: str,
    PaymentMethod: str,
    PaperlessBilling: str,
    ContentType: str,
    MultiDeviceAccess: str,
    DeviceRegistered: str,
    ViewingHoursPerWeek: float,
    AverageViewingDuration: float,
    ContentDownloadsPerMonth: int,
    GenrePreference: str,
    UserRating: float,
    SupportTicketsPerMonth: int,
    Gender: str,
    WatchlistSize: int,
    ParentalControl: str,
    SubtitlesEnabled: str
):

    X_pred = pd.DataFrame(dict(
        AccountAge=[AccountAge],
        MonthlyCharges=[MonthlyCharges],
        TotalCharges=[TotalCharges],
        SubscriptionType=[SubscriptionType],
        PaymentMethod=[PaymentMethod],
        PaperlessBilling=[PaperlessBilling],
        ContentType=[ContentType],
        MultiDeviceAccess=[MultiDeviceAccess],
        DeviceRegistered=[DeviceRegistered],
        ViewingHoursPerWeek=[ViewingHoursPerWeek],
        AverageViewingDuration=[AverageViewingDuration],
        ContentDownloadsPerMonth=[ContentDownloadsPerMonth],
        GenrePreference=[GenrePreference],
        UserRating=[UserRating],
        SupportTicketsPerMonth=[SupportTicketsPerMonth],
        Gender=[Gender],
        WatchlistSize=[WatchlistSize],
        ParentalControl=[ParentalControl],
        SubtitlesEnabled=[SubtitlesEnabled]
    ))

    test_payload = pd.DataFrame(X_pred)
    prediction, probability = pred(X_pred)

    print(prediction)
    print(probability)
    return {
        "churn_prediction": int(prediction[0]),
        "churn_probability": float(probability[0])
    }
    #return{
    #     "churn_prediction": 8,
    #     "churn_probability": 9
    #}
