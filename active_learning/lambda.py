
from io import StringIO
import predict
import pandas

class Duck:
    pass

def lambda_handler(event, context):
    args = Duck()
    args.model = "training/all_crops_all_diagnostics_2/catboost/model.pkl"
    model = predict.load_model(args)
    input_df = pandas.read_csv(StringIO(event['input']))
    result_df = predict.predict(model, input_df)
    result_csv = StringIO()
    result_df.to_csv(result_csv)
    return {
        "statusCode": 200,
        "message": str(result_csv)
    }