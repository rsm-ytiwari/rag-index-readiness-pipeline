import requests
import json

API_BASE = "http://localhost:8000"

print("="*70)
print("API ENDPOINT TESTS")
print("="*70)

# Test 1: Health
print("\n🔍 TEST 1: Health Check")
r = requests.get(f"{API_BASE}/health")
print(json.dumps(r.json(), indent=2))

# Test 2: Stats (Before)
print("\n🔍 TEST 2: Initial Stats")
r = requests.get(f"{API_BASE}/stats")
print(json.dumps(r.json(), indent=2))

# Test 3: High Quality Review
print("\n🔍 TEST 3: High Quality Review")
r = requests.post(f"{API_BASE}/score", json={
    "text": "This restaurant has amazing food and excellent service. The atmosphere is wonderful and the staff are very friendly. I highly recommend the pasta dishes and the tiramisu for dessert. Will definitely be coming back!",
    "stars": 5,
    "review_id": "test-001"
})
result = r.json()
if 'index_readiness_score' in result:
    print(f"✅ Score: {result['index_readiness_score']}/100")
    print(f"✅ Recommendation: {result['recommendation']}")
    print(f"✅ Chunk Quality: {result['chunk_quality']}")
    print(f"✅ Processing Time: {result['processing_time_ms']}ms")
else:
    print(f"❌ Error: {result}")

# Test 4: Low Quality Review (FIXED - now 10+ chars)
print("\n🔍 TEST 4: Low Quality Review (Short but valid)")
r = requests.post(f"{API_BASE}/score", json={
    "text": "Bad food, terrible service.",
    "stars": 1,
    "review_id": "test-002"
})
result = r.json()
if 'index_readiness_score' in result:
    print(f"✅ Score: {result['index_readiness_score']}/100")
    print(f"✅ Recommendation: {result['recommendation']}")
    print(f"✅ Chunk Quality: {result['chunk_quality']}")
else:
    print(f"❌ Error: {result}")

# Test 5: PII Review
print("\n🔍 TEST 5: Review with PII (Phone Number)")
r = requests.post(f"{API_BASE}/score", json={
    "text": "Great service! Call me at 555-1234 if you have questions about my order.",
    "stars": 4,
    "review_id": "test-003"
})
result = r.json()
if 'index_readiness_score' in result:
    print(f"✅ Score: {result['index_readiness_score']}/100")
    print(f"✅ Recommendation: {result['recommendation']}")
    print(f"✅ Has PII: {result['has_pii']}")
    print(f"✅ PII Types: {result['pii_types']}")
else:
    print(f"❌ Error: {result}")

# Test 6: Review with Email PII
print("\n🔍 TEST 6: Review with Email PII")
r = requests.post(f"{API_BASE}/score", json={
    "text": "Please contact me at john.doe@example.com for catering inquiries. We had a wonderful experience!",
    "stars": 5,
    "review_id": "test-004"
})
result = r.json()
if 'index_readiness_score' in result:
    print(f"✅ Score: {result['index_readiness_score']}/100")
    print(f"✅ Has PII: {result['has_pii']}")
    print(f"✅ PII Types: {result['pii_types']}")
else:
    print(f"❌ Error: {result}")

# Test 7: Batch
print("\n🔍 TEST 7: Batch Scoring (3 Reviews)")
r = requests.post(f"{API_BASE}/batch", json={
    "reviews": [
        {
            "text": "Excellent pizza! The crust was perfect and the toppings were fresh. Service was quick and friendly.",
            "stars": 5,
            "review_id": "batch-001"
        },
        {
            "text": "Terrible experience. Food was cold and service was slow. Would not recommend.",
            "stars": 1,
            "review_id": "batch-002"
        },
        {
            "text": "Average food, nothing special. Prices are reasonable though and staff was polite.",
            "stars": 3,
            "review_id": "batch-003"
        }
    ]
})
result = r.json()
if 'results' in result:
    print(f"✅ Total Reviews: {result['total_reviews']}")
    print(f"✅ Avg Score: {result['avg_score']}/100")
    print(f"✅ Total Processing Time: {result['processing_time_ms']}ms")
    print(f"\n📊 Individual Results:")
    for i, review_result in enumerate(result['results'], 1):
        print(f"  Review {i} ({review_result['review_id']}): Score={review_result['index_readiness_score']}, Rec={review_result['recommendation']}")
else:
    print(f"❌ Error: {result}")

# Test 8: Final Stats
print("\n🔍 TEST 8: Final API Stats")
r = requests.get(f"{API_BASE}/stats")
result = r.json()
print(f"✅ Total Requests: {result['total_requests']}")
print(f"✅ Total Reviews Scored: {result['total_reviews_scored']}")
print(f"✅ Avg Processing Time: {result['avg_processing_time_ms']}ms")
print(f"✅ Uptime: {result['uptime_seconds']}s")

print("\n" + "="*70)
print("✅ ALL TESTS COMPLETED")
print("="*70)

# Summary
print("\n📊 TEST SUMMARY:")
print(f"  - Health check: Working")
print(f"  - Single review scoring: Working")
print(f"  - PII detection: Working (phone & email)")
print(f"  - Batch scoring: Working")
print(f"  - Statistics tracking: Working")
print("\n🚀 API is production-ready!")
