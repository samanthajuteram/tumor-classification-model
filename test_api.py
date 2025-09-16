import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    try:
        response = requests.get(f"{BASE_URL}/")
        print("Health Check:")
        print(json.dumps(response.json(), indent=2))
        print()
        return True
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_model_info():
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        print("Model Info:")
        print(json.dumps(response.json(), indent=2))
        print()
        return True
    except Exception as e:
        print(f"Model info failed: {e}")
        return False

def test_single_prediction():
    # Sample data
    sample_data = {
        "concave_points_mean": 0.05,
        "area_worst": 800.0,
        "fractal_dimension_worst": 0.08,
        "smoothness_worst": 0.15,
        "symmetry_worst": 0.25,
        "fractal_dimension_mean": 0.06,
        "smoothness_mean": 0.1,
        "compactness_se": 0.02,
        "fractal_dimension_se": 0.003,
        "concave_points_se": 0.01,
        "symmetry_se": 0.015,
        "perimeter_se": 3.0,
        "concavity_se": 0.02,
        "symmetry_mean": 0.18,
        "smoothness_se": 0.005,
        "texture_se": 1.5,
        "texture_mean": 15.0
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=sample_data)
        print("Single Prediction:")
        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
            print(f" Prediction: {result['prediction']}")
            print(f" Probability: {result['probability']}")
            print(f" Confidence: {result['confidence']}")
        else:
            print(f" Error {response.status_code}: {response.text}")
        print()
        return response.status_code == 200
    except Exception as e:
        print(f"Single prediction failed: {e}")
        return False

def test_batch_prediction():
    # Sample batch data (2 patients)
    batch_data = [
        {
            "concave_points_mean": 0.05,
            "area_worst": 800.0,
            "fractal_dimension_worst": 0.08,
            "smoothness_worst": 0.15,
            "symmetry_worst": 0.25,
            "fractal_dimension_mean": 0.06,
            "smoothness_mean": 0.1,
            "compactness_se": 0.02,
            "fractal_dimension_se": 0.003,
            "concave_points_se": 0.01,
            "symmetry_se": 0.015,
            "perimeter_se": 3.0,
            "concavity_se": 0.02,
            "symmetry_mean": 0.18,
            "smoothness_se": 0.005,
            "texture_se": 1.5,
            "texture_mean": 15.0
        },
        {
            "concave_points_mean": 0.1,
            "area_worst": 1200.0,
            "fractal_dimension_worst": 0.12,
            "smoothness_worst": 0.2,
            "symmetry_worst": 0.3,
            "fractal_dimension_mean": 0.08,
            "smoothness_mean": 0.15,
            "compactness_se": 0.05,
            "fractal_dimension_se": 0.006,
            "concave_points_se": 0.03,
            "symmetry_se": 0.02,
            "perimeter_se": 5.0,
            "concavity_se": 0.04,
            "symmetry_mean": 0.22,
            "smoothness_se": 0.008,
            "texture_se": 2.0,
            "texture_mean": 20.0
        }
    ]
    
    try:
        response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
        print("Batch Prediction:")
        if response.status_code == 200:
            result = response.json()
            print(f" Processed {result['total_patients']} patients")
            for pred in result['predictions']:
                print(f"   Patient {pred['patient_id']}: {pred['prediction']} ({pred['confidence']} confidence)")
        else:
            print(f" Error {response.status_code}: {response.text}")
        print()
        return response.status_code == 200
    except Exception as e:
        print(f"Batch prediction failed: {e}")
        return False

def test_features_info():
    try:
        response = requests.get(f"{BASE_URL}/features")
        print("Features Info:")
        result = response.json()
        print(f" Total features required: {result['total_features']}")
        print(" Feature list retrieved successfully")
        print()
        return True
    except Exception as e:
        print(f"Features info failed: {e}")
        return False

def run_all_tests():
    """Run all API tests"""
    print("Testing Breast Cancer Diagnosis API")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Model Info", test_model_info),
        ("Features Info", test_features_info),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f" Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f" {test_name} PASSED")
            else:
                print(f" {test_name} FAILED")
        except Exception as e:
            print(f" {test_name} FAILED: {e}")
        print("-" * 30)
    
    print("\n" + "=" * 50)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print(" All tests passed! Your API is working correctly.")
    else:
        print(" Some tests failed. Check the API and model setup.")
    
    return passed == total

if __name__ == "__main__":
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print(" Error: Could not connect to API.")
        print(" Make sure the API server is running:")
        print("   1. Open a terminal and run: python api.py")
        print("   2. Then run this test script in another terminal")
    except Exception as e:
        print(f" Unexpected error: {e}")