import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .ml_loader import load_feature_names, predict_single

AVAILABLE_MODELS = [
    {
        "file": "binary_rf.joblib",
        "task": "Binary",
        "algorithm": "Random Forest",
        "output": "BENIGN / ATTACK",
    },
    {
        "file": "multiclass_rf.joblib",
        "task": "Multiclass",
        "algorithm": "Random Forest",
        "output": "13 classes",
    },
]


def home(request):
    return render(request, "detector/home.html", {"models": AVAILABLE_MODELS})


def manual_predict(request):
    feature_names = list(load_feature_names())
    fields = [
        {"index": i, "name": name}
        for i, name in enumerate(feature_names)
    ]
    context = {
        "fields": fields,
        "task": "binary",
        "error": None,
    }

    if request.method == "POST":
        task = (request.POST.get("task") or "binary").strip().lower()
        context["task"] = task
        try:
            features = []
            for i, name in enumerate(feature_names):
                raw = request.POST.get(f"feat_{i}", "").strip()
                if raw == "":
                    raise ValueError(f"Valeur manquante pour le champ '{name}' (feat_{i}).")
                features.append(float(raw))
            print(f"[DEBUG manual_predict] Processing prediction: task='{task}', features_count={len(features)}")
            result = predict_single(features, task)
            print(f"[DEBUG manual_predict] Prediction successful: {result}")
        except Exception as exc:
            print(f"[ERROR manual_predict] Exception during prediction: {exc}")
            context["error"] = str(exc)
        else:
            return render(request, "detector/result.html", {"result": result})

    return render(request, "detector/form.html", context)


@csrf_exempt
def api_predict(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST is allowed."}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON."}, status=400)

    features = payload.get("features")
    task = payload.get("task", "multiclass")
    if not isinstance(features, list):
        return JsonResponse({"error": "'features' must be a list of numbers."}, status=400)

    try:
        values = [float(v) for v in features]
        result = predict_single(values, task)
    except (TypeError, ValueError) as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    return JsonResponse(result)
