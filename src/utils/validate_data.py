# import great_expectations as ge
# from typing import Tuple, List


# def validate_telco_data(df) -> Tuple[bool, List[str]]:
    
#     print("Starting data validation with Great Expectations...")
    
   
#     ge_df = ge.dataset.PandasDataset(df)
    
    
#     print("Validating schema and required columns...")
    
    
#     ge_df.expect_column_to_exist("customerID")
#     ge_df.expect_column_values_to_not_be_null("customerID")
    
    
#     ge_df.expect_column_to_exist("gender") 
#     ge_df.expect_column_to_exist("Partner")
#     ge_df.expect_column_to_exist("Dependents")
    
    
#     ge_df.expect_column_to_exist("PhoneService")
#     ge_df.expect_column_to_exist("InternetService")
#     ge_df.expect_column_to_exist("Contract")
    
    
#     ge_df.expect_column_to_exist("tenure")
#     ge_df.expect_column_to_exist("MonthlyCharges")
#     ge_df.expect_column_to_exist("TotalCharges")
    
   
#     print("Validating business logic constraints...")
    
    
#     ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"])
    
    
#     ge_df.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
#     ge_df.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
#     ge_df.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])
    
    
#     ge_df.expect_column_values_to_be_in_set(
#         "Contract", 
#         ["Month-to-month", "One year", "Two year"]
#     )
    
    
#     ge_df.expect_column_values_to_be_in_set(
#         "InternetService",
#         ["DSL", "Fiber optic", "No"]
#     )
    
#     print("Validating numeric ranges and business constraints...")
    
   
#     ge_df.expect_column_values_to_be_between("tenure", min_value=0)
    
   
#     ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0)
    
  
#     ge_df.expect_column_values_to_be_between("TotalCharges", min_value=0)
    
    
#     print("Validating statistical properties...")
    
    
#     ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)
    
    
#     ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200)
    
#     ge_df.expect_column_values_to_not_be_null("tenure")
#     ge_df.expect_column_values_to_not_be_null("MonthlyCharges")
    
    
#     print("Validating data consistency...")
    
    
#     ge_df.expect_column_pair_values_A_to_be_greater_than_B(
#         column_A="TotalCharges",
#         column_B="MonthlyCharges",
#         or_equal=True,
#         mostly=0.95  
#     )
    
 
#     print("Running complete validation suite...")
#     results = ge_df.validate()
    
    
#     failed_expectations = []
#     for r in results["results"]:
#         if not r["success"]:
#             expectation_type = r["expectation_config"]["expectation_type"]
#             failed_expectations.append(expectation_type)
    
#     # Print validation summary
#     total_checks = len(results["results"])
#     passed_checks = sum(1 for r in results["results"] if r["success"])
#     failed_checks = total_checks - passed_checks
    
#     if results["success"]:
#         print(f" Data validation PASSED: {passed_checks}/{total_checks} checks successful")
#     else:
#         print(f" Data validation FAILED: {failed_checks}/{total_checks} checks failed")
#         print(f"   Failed expectations: {failed_expectations}")
    
#     return results["success"], failed_expectations


import pandas as pd
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    print("Starting data validation...")
    failed_expectations = []
    total_checks = 0

    def check(name, condition):
        total_checks_ref.append(1)
        if not condition:
            failed_expectations.append(name)

    total_checks_ref = []

    # Required columns
    required_columns = [
        "customerID", "gender", "Partner", "Dependents",
        "PhoneService", "InternetService", "Contract",
        "tenure", "MonthlyCharges", "TotalCharges"
    ]

    print("Validating schema and required columns...")
    for col in required_columns:
        check(f"expect_column_to_exist:{col}", col in df.columns)

    # Stop early if columns are missing
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        print(f" Data validation FAILED: missing columns {missing}")
        return False, failed_expectations

    print("Validating business logic constraints...")
    check("expect_column_values_to_not_be_null:customerID",
          df["customerID"].notnull().all())

    check("expect_column_values_to_be_in_set:gender",
          df["gender"].isin(["Male", "Female"]).all())

    check("expect_column_values_to_be_in_set:Partner",
          df["Partner"].isin(["Yes", "No"]).all())

    check("expect_column_values_to_be_in_set:Dependents",
          df["Dependents"].isin(["Yes", "No"]).all())

    check("expect_column_values_to_be_in_set:PhoneService",
          df["PhoneService"].isin(["Yes", "No"]).all())

    check("expect_column_values_to_be_in_set:Contract",
          df["Contract"].isin(["Month-to-month", "One year", "Two year"]).all())

    check("expect_column_values_to_be_in_set:InternetService",
          df["InternetService"].isin(["DSL", "Fiber optic", "No"]).all())

    print("Validating numeric ranges and business constraints...")
    
# Convert TotalCharges to numeric (it loads as string in raw Telco data)
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    check("expect_column_values_to_not_be_null:tenure",
        df["tenure"].notnull().all())
    

    check("expect_column_values_to_not_be_null:MonthlyCharges",
          df["MonthlyCharges"].notnull().all())

    check("expect_column_values_to_be_between:tenure",
          ((df["tenure"] >= 0) & (df["tenure"] <= 120)).all())

    check("expect_column_values_to_be_between:MonthlyCharges",
          ((df["MonthlyCharges"] >= 0) & (df["MonthlyCharges"] <= 200)).all())

    check("expect_column_values_to_be_between:TotalCharges",
      (df["TotalCharges"].dropna() >= 0).all())

    print("Validating data consistency...")
    pair_check = (df["TotalCharges"] >= df["MonthlyCharges"]).mean() >= 0.95
    check("expect_column_pair_values_A_to_be_greater_than_B:TotalCharges_MonthlyCharges",
          pair_check)

    total = len(total_checks_ref)
    passed = total - len(failed_expectations)
    is_valid = len(failed_expectations) == 0

    if is_valid:
        print(f" Data validation PASSED: {passed}/{total} checks successful")
    else:
        print(f" Data validation FAILED: {len(failed_expectations)}/{total} checks failed")
        print(f"   Failed expectations: {failed_expectations}")

    return is_valid, failed_expectations