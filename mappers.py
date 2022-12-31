def map_to_model(mongo_record):
    return {
        "type": mongo_record["type"],
        "name": mongo_record["name"],
        "accuracy_score": mongo_record["accuracy_score"],
        "split_ratio": mongo_record["split_ratio"],
        "f1_score": mongo_record["f1_score"],
        "uuid": mongo_record["uuid"],
        "size": mongo_record["size"],
        "active": mongo_record["active"],
        "input_variables": mongo_record["input_variables"],
        "output_variables": mongo_record["output_variables"],
        "physical_id": str(mongo_record["_id"]),
        "details": mongo_record["details"]
    }
