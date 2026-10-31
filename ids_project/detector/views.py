from django.shortcuts import render

def home(request):
    result = None
    if request.method == 'POST':
        flow_duration = float(request.POST.get('flow_duration', 0))
        packets_rate = float(request.POST.get('flow_packets_s', 0))

        is_attack = flow_duration > 10000 and packets_rate > 300
        
        result = {
            'is_attack': is_attack,
            'label': 'Attaque Détectée (DDoS / DoS)' if is_attack else 'BENIGN',
            'confidence': 94.2 if is_attack else 98.7
        }

    return render(request, 'detector/index.html', {'result': result})