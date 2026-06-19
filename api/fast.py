import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from retention_agent.interface.main import pred
from retention_agent.interface.main_kkbox import pred_kkbox

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

    prediction, probability = pred(X_pred)

    return {
        "churn_prediction": int(prediction[0]),
        "churn_probability": float(probability[0])
    }

@app.get("/predict_kkbox")
def predict_kkbox(
    city: int,
    bd_clean: float,
    bd_was_invalid: int,
    gender_filled: str,
    registered_via: int,
    tenure_days: float,
    n_transactions: float,
    n_cancels_before_cutoff: float,
    mean_actual_paid: float,
    sum_actual_paid: float,
    mean_list_price: float,
    mean_plan_days: float,
    mean_auto_renew: float,
    n_unique_payment_methods: float,
    discount_ratio: float,
    days_since_last_txn: float,
    days_until_expiry_at_cutoff: float,
    latest_payment_method_id: float,
):
    X_pred = pd.DataFrame(dict(
        city=[city],
        bd_clean=[bd_clean],
        bd_was_invalid=[bd_was_invalid],
        gender_filled=[gender_filled],
        registered_via=[registered_via],
        tenure_days=[tenure_days],
        n_transactions=[n_transactions],
        n_cancels_before_cutoff=[n_cancels_before_cutoff],
        mean_actual_paid=[mean_actual_paid],
        sum_actual_paid=[sum_actual_paid],
        mean_list_price=[mean_list_price],
        mean_plan_days=[mean_plan_days],
        mean_auto_renew=[mean_auto_renew],
        n_unique_payment_methods=[n_unique_payment_methods],
        discount_ratio=[discount_ratio],
        days_since_last_txn=[days_since_last_txn],
        days_until_expiry_at_cutoff=[days_until_expiry_at_cutoff],
        latest_payment_method_id=[latest_payment_method_id],
    ))

    prediction, probability, _ = pred_kkbox(X_pred)

    return {
        "churn_prediction": int(prediction[0]),
        "churn_probability": float(probability[0]),
    }


@app.get("/explain_kkbox")
def explain_kkbox(
    city: int,
    bd_clean: float,
    bd_was_invalid: int,
    gender_filled: str,
    registered_via: int,
    tenure_days: float,
    n_transactions: float,
    n_cancels_before_cutoff: float,
    mean_actual_paid: float,
    sum_actual_paid: float,
    mean_list_price: float,
    mean_plan_days: float,
    mean_auto_renew: float,
    n_unique_payment_methods: float,
    discount_ratio: float,
    days_since_last_txn: float,
    days_until_expiry_at_cutoff: float,
    latest_payment_method_id: float,
):
    X_pred = pd.DataFrame(dict(
        city=[city],
        bd_clean=[bd_clean],
        bd_was_invalid=[bd_was_invalid],
        gender_filled=[gender_filled],
        registered_via=[registered_via],
        tenure_days=[tenure_days],
        n_transactions=[n_transactions],
        n_cancels_before_cutoff=[n_cancels_before_cutoff],
        mean_actual_paid=[mean_actual_paid],
        sum_actual_paid=[sum_actual_paid],
        mean_list_price=[mean_list_price],
        mean_plan_days=[mean_plan_days],
        mean_auto_renew=[mean_auto_renew],
        n_unique_payment_methods=[n_unique_payment_methods],
        discount_ratio=[discount_ratio],
        days_since_last_txn=[days_since_last_txn],
        days_until_expiry_at_cutoff=[days_until_expiry_at_cutoff],
        latest_payment_method_id=[latest_payment_method_id],
    ))

    _, _, top_drivers = pred_kkbox(X_pred)

    return {
        "top_drivers": top_drivers,
    }
