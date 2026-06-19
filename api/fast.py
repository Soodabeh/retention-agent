import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from retention_agent.interface.main import pred

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
        return dict(greeting="Hello")

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
    # 💡 Optional trick instead of writing each column name manually:
    # locals() gets us all of our arguments back as a dictionary
    # https://docs.python.org/3/library/functions.html#locals
    # X_pred = pd.DataFrame(locals(), index=[0])

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

    prediction, probability = pred(X_pred)

    return {
        "churn_prediction": int(prediction[0]),
        "churn_probability": float(probability[0])
    }

@app.get("/predict_with_defaults")
def predict_with_defaults(
  AccountAge: int = Query(24),
    MonthlyCharges: float = Query(79.99),
    TotalCharges: float = Query(1919.76),
    SubscriptionType: str = Query("Basic"),
    PaymentMethod: str = Query("Mailed check"),
    PaperlessBilling: str = Query("No"),
    ContentType: str = Query("Movies"),
    MultiDeviceAccess: str = Query("Yes"),
    DeviceRegistered: str = Query("Computer"),
    ViewingHoursPerWeek: float = Query(15.5),
    AverageViewingDuration: float = Query(120.0),
    ContentDownloadsPerMonth: int = Query(5),
    GenrePreference: str = Query("Action"),
    UserRating: float = Query(4.5),
    SupportTicketsPerMonth: int = Query(1),
    Gender: str = Query("Female"),
    WatchlistSize: float = Query(12.0),
    ParentalControl: str = Query("No"),
    SubtitlesEnabled: str = Query("Yes")
):
    # This creates the DataFrame using the values provided, or falling back to your static data
    X_pred = pd.DataFrame(dict(
        AccountAge=[int(AccountAge)],
        MonthlyCharges=[float(MonthlyCharges)],
        TotalCharges=[float(TotalCharges)],
        SubscriptionType=[SubscriptionType],
        PaymentMethod=[PaymentMethod],
        PaperlessBilling=[PaperlessBilling],
        ContentType=[ContentType],
        MultiDeviceAccess=[MultiDeviceAccess],
        DeviceRegistered=[DeviceRegistered],
        ViewingHoursPerWeek=[float(ViewingHoursPerWeek)],
        AverageViewingDuration=[float(AverageViewingDuration)],
        ContentDownloadsPerMonth=[int(ContentDownloadsPerMonth)],
        GenrePreference=[GenrePreference],
        UserRating=[float(UserRating)],
        SupportTicketsPerMonth=[int(SupportTicketsPerMonth)],
        Gender=[Gender],
        WatchlistSize=[int(WatchlistSize)],
        ParentalControl=[ParentalControl],
        SubtitlesEnabled=[SubtitlesEnabled]
    ))

    prediction, probability = pred(X_pred)

    return {
        "churn_prediction": int(prediction[0]),
        "churn_probability": float(probability[0])
    }
