"""
Detector de Deepfakes - Versão com Correções Essenciais

"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.fftpack import fft2, fftfreq
import matplotlib.pyplot as plt
import os
from pathlib import Path
import json
from datetime import datetime

class DeepfakeDetector:
    def __init__(self):
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("✅ OpenCV carregado!")
        except Exception as e:
            print(f"❌ Erro OpenCV: {e}")
            return
            
        # Thresholds corrigidos - valores mais baixos e sensíveis
        self.thresholds = {
            'edge_inconsistency': 0.12,      # era 0.25
            'spectral_anomaly': 0.18,        # era 0.35
            'color_variance': 0.15,          # era 0.2
            'temporal_flicker': 0.08,        # era 0.1
            'optical_flow_anomaly': 0.25     # era 0.3
        }
        
        self.frame_history = []
        self.max_history = 10
        
    def detect_roi(self, frame):
        if frame is None:
            return None, None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None, None
            
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        margin = int(0.2 * min(w, h))
        x_roi = max(0, x - margin)
        y_roi = max(0, y - margin)
        w_roi = min(frame.shape[1] - x_roi, w + 2 * margin)
        h_roi = min(frame.shape[0] - y_roi, h + 2 * margin)
        
        roi = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        return roi, (x_roi, y_roi, w_roi, h_roi)
    
    def analyze_edges(self, roi):
        if roi is None:
            return {'edge_score': 0, 'details': 'ROI não encontrada'}
            
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        canny_edges = cv2.Canny(gray_roi, 50, 150)
        
        sobel_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        edge_density = np.sum(canny_edges > 0) / (canny_edges.shape[0] * canny_edges.shape[1])
        edge_variance = np.var(sobel_magnitude)
        
        # Score corrigido - melhor normalização
        edge_score = min((edge_density * 3.0) + (edge_variance / 8000.0), 1.0)
        
        details = {
            'edge_density': float(edge_density),
            'edge_variance': float(edge_variance),
            'suspicious': edge_score > self.thresholds['edge_inconsistency']
        }
        
        return {'edge_score': float(edge_score), 'details': details}
    
    def analyze_spectral(self, roi):
        if roi is None:
            return {'spectral_score': 0, 'details': 'ROI não encontrada'}
            
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float64)
        
        # Normalização melhorada para evitar valores zerados
        gray_roi = (gray_roi - np.mean(gray_roi)) / (np.std(gray_roi) + 1e-6)
        
        f_transform = fft2(gray_roi)
        f_magnitude = np.abs(f_transform)
        
        h, w = gray_roi.shape
        
        # Regiões de frequência corrigidas
        low_freq_energy = np.sum(f_magnitude[:h//8, :w//8] ** 2)
        high_freq_energy = np.sum(f_magnitude[7*h//8:, 7*w//8:] ** 2)
        total_energy = np.sum(f_magnitude ** 2)
        
        # Evita divisão por zero
        if total_energy > 1e-10:
            freq_ratio = high_freq_energy / (low_freq_energy + 1e-10)
        else:
            freq_ratio = 0.0
            
        # DCT corrigido
        try:
            dct_coeffs = cv2.dct(gray_roi.astype(np.float32))
            dct_variance = np.var(dct_coeffs)
        except:
            dct_variance = 0.0
        
        # Score corrigido - melhor normalização
        spectral_score = min(abs(0.1 - freq_ratio) + (dct_variance * 100.0), 1.0)
        
        # Interpretações
        freq_interpretation = ""
        if freq_ratio < 0.01:
            freq_interpretation = "Muito baixa - possível suavização excessiva"
        elif freq_ratio > 1.0:
            freq_interpretation = "Muito alta - possível ruído artificial"
        else:
            freq_interpretation = "Normal"
            
        dct_interpretation = ""
        if dct_variance < 0.001:
            dct_interpretation = "Muito baixa - possível compressão artificial"
        elif dct_variance > 0.01:
            dct_interpretation = "Muito alta - possível artefatos de edição"
        else:
            dct_interpretation = "Normal"
        
        details = {
            'freq_ratio': float(freq_ratio),
            'freq_interpretation': freq_interpretation,
            'dct_variance': float(dct_variance),
            'dct_interpretation': dct_interpretation,
            'low_freq_energy': float(low_freq_energy),
            'high_freq_energy': float(high_freq_energy),
            'suspicious': spectral_score > self.thresholds['spectral_anomaly']
        }
        
        return {'spectral_score': float(spectral_score), 'details': details}
    
    def analyze_color_statistics(self, roi):
        if roi is None:
            return {'color_score': 0, 'details': 'ROI não encontrada'}
            
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        h, w = roi.shape[:2]
        center_region = roi[h//4:3*h//4, w//4:3*w//4]
        
        stats = {}
        total_inconsistency = 0
        
        for i, channel in enumerate(['B', 'G', 'R']):
            roi_channel = roi[:, :, i]
            center_channel = center_region[:, :, i]
            
            roi_mean = np.mean(roi_channel)
            roi_std = np.std(roi_channel)
            center_mean = np.mean(center_channel)
            center_std = np.std(center_channel)
            
            mean_diff = abs(roi_mean - center_mean)
            std_ratio = center_std / (roi_std + 1e-8)
            
            # Inconsistência corrigida
            channel_inconsistency = (mean_diff / 255.0) + abs(1.0 - std_ratio) * 0.5
            total_inconsistency += channel_inconsistency
            
            interpretation = ""
            if mean_diff > 15:
                interpretation = "Diferença significativa centro-borda"
            elif abs(1.0 - std_ratio) > 0.3:
                interpretation = "Variação de textura anômala"
            else:
                interpretation = "Normal"
            
            stats[channel] = {
                'roi_mean': float(roi_mean),
                'roi_std': float(roi_std),
                'center_mean': float(center_mean),
                'center_std': float(center_std),
                'mean_diff': float(mean_diff),
                'std_ratio': float(std_ratio),
                'inconsistency': float(channel_inconsistency),
                'interpretation': interpretation
            }
        
        # Score normalizado
        color_score = min(total_inconsistency / 3.0, 1.0)
        
        # Análise de saturação
        saturation = hsv_roi[:, :, 1]
        sat_mean = np.mean(saturation)
        sat_std = np.std(saturation)
        
        sat_interpretation = ""
        if sat_mean < 40:
            sat_interpretation = "Muito baixa - possível dessaturação artificial"
        elif sat_mean > 200:
            sat_interpretation = "Muito alta - possível saturação artificial"
        else:
            sat_interpretation = "Normal"
        
        details = {
            'channel_stats': stats,
            'average_inconsistency': float(color_score),
            'saturation_mean': float(sat_mean),
            'saturation_std': float(sat_std),
            'saturation_interpretation': sat_interpretation,
            'suspicious': color_score > self.thresholds['color_variance']
        }
        
        return {'color_score': float(color_score), 'details': details}
    
    def apply_median_filter_analysis(self, roi):
        if roi is None:
            return {'residual_score': 0, 'details': 'ROI não encontrada'}
            
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        median_filtered = cv2.medianBlur(gray_roi, 5)
        
        residual = np.abs(gray_roi.astype(np.float32) - median_filtered.astype(np.float32))
        
        residual_mean = np.mean(residual)
        residual_std = np.std(residual)
        
        # Score corrigido
        residual_score = min((residual_std / 15.0) + (residual_mean / 20.0), 1.0)
        
        details = {
            'residual_mean': float(residual_mean),
            'residual_std': float(residual_std),
            'suspicious': residual_score > 0.08  # threshold mais baixo
        }
        
        return {'residual_score': float(residual_score), 'details': details}
    
    def analyze_temporal_consistency(self, current_frame):
        if current_frame is None:
            return {'temporal_score': 0, 'details': 'Frame inválido'}
            
        self.frame_history.append(current_frame)
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)
            
        if len(self.frame_history) < 2:
            return {'temporal_score': 0, 'details': 'Histórico insuficiente'}
        
        temporal_scores = []
        
        for i in range(1, len(self.frame_history)):
            prev_roi, _ = self.detect_roi(self.frame_history[i-1])
            curr_roi, _ = self.detect_roi(self.frame_history[i])
            
            if prev_roi is None or curr_roi is None:
                continue
                
            target_size = (100, 100)
            prev_resized = cv2.resize(prev_roi, target_size)
            curr_resized = cv2.resize(curr_roi, target_size)
            
            frame_diff = np.abs(curr_resized.astype(np.float32) - prev_resized.astype(np.float32))
            flicker_score = np.mean(frame_diff) / 255.0
            temporal_scores.append(flicker_score)
        
        if temporal_scores:
            avg_temporal_score = np.mean(temporal_scores)
            temporal_variance = np.var(temporal_scores)
        else:
            avg_temporal_score = 0
            temporal_variance = 0
            
        details = {
            'avg_flicker': float(avg_temporal_score),
            'temporal_variance': float(temporal_variance),
            'frames_analyzed': len(temporal_scores),
            'suspicious': avg_temporal_score > self.thresholds['temporal_flicker']
        }
        
        return {'temporal_score': float(avg_temporal_score), 'details': details}
    
    def convert_to_json_serializable(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    def analyze_frame(self, frame):
        roi, roi_coords = self.detect_roi(frame)
        
        if roi is None:
            return {
                'edge_analysis': {'edge_score': 0, 'details': 'ROI não encontrada'},
                'spectral_analysis': {'spectral_score': 0, 'details': 'ROI não encontrada'},
                'color_analysis': {'color_score': 0, 'details': 'ROI não encontrada'},
                'residual_analysis': {'residual_score': 0, 'details': 'ROI não encontrada'},
                'temporal_analysis': {'temporal_score': 0, 'details': 'ROI não encontrada'}
            }
        
        edge_analysis = self.analyze_edges(roi)
        spectral_analysis = self.analyze_spectral(roi)
        color_analysis = self.analyze_color_statistics(roi)
        residual_analysis = self.apply_median_filter_analysis(roi)
        temporal_analysis = self.analyze_temporal_consistency(frame)
        
        return {
            'edge_analysis': edge_analysis,
            'spectral_analysis': spectral_analysis,
            'color_analysis': color_analysis,
            'residual_analysis': residual_analysis,
            'temporal_analysis': temporal_analysis,
            'roi_coords': roi_coords
        }
    
    def generate_report(self, analyses):
        total_score = 0
        suspicious_indicators = 0
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_assessment': '',
            'confidence_level': 0,
            'detailed_analysis': {},
            'recommendations': []
        }
        
        for analysis_name, result in analyses.items():
            if isinstance(result, dict):
                score = result.get('score', 0)
                details = result.get('details', {})
            else:
                score = 0
                details = {}
            
            score = float(score) if score else 0.0
            total_score += score
            
            if isinstance(details, dict) and details.get('suspicious', False):
                suspicious_indicators += 1
                
            report['detailed_analysis'][analysis_name] = {
                'score': float(score),
                'status': 'SUSPEITO' if (isinstance(details, dict) and details.get('suspicious', False)) else 'NORMAL',
                'details': self.convert_to_json_serializable(details)
            }
        
        avg_score = float(total_score / len(analyses)) if analyses else 0.0
        confidence = float(suspicious_indicators / len(analyses)) if analyses else 0.0
        
        if suspicious_indicators >= 3:
            assessment = 'ALTAMENTE SUSPEITO - Múltiplos indicadores de manipulação'
            recommendations = [
                'Verificar com especialista em forense digital',
                'Analisar metadados do arquivo',
                'Comparar com outras fontes do mesmo conteúdo'
            ]
        elif suspicious_indicators >= 1:
            assessment = 'MODERADAMENTE SUSPEITO - Alguns indicadores anômalos'
            recommendations = [
                'Verificação adicional recomendada',
                'Comparar qualidade com vídeos similares da mesma fonte'
            ]
        else:
            assessment = 'PROVAVELMENTE AUTÊNTICO - Nenhum indicador significativo'
            recommendations = [
                'Vídeo parece consistente com padrões normais',
                'Verificação de rotina suficiente'
            ]
            
        report['overall_assessment'] = assessment
        report['confidence_level'] = float(confidence)
        report['recommendations'] = recommendations
        
        return report
    
    def aggregate_frame_analyses(self, frame_analyses):
        aggregated = {}
        
        analysis_types = ['edge_analysis', 'spectral_analysis', 'color_analysis', 
                         'residual_analysis', 'temporal_analysis']
        
        for analysis_type in analysis_types:
            scores = []
            all_suspicious = []
            
            for frame_analysis in frame_analyses:
                if analysis_type in frame_analysis and isinstance(frame_analysis[analysis_type], dict):
                    analysis_data = frame_analysis[analysis_type]
                    
                    score = 0
                    if 'edge_score' in analysis_data:
                        score = analysis_data['edge_score']
                    elif 'spectral_score' in analysis_data:
                        score = analysis_data['spectral_score']
                    elif 'color_score' in analysis_data:
                        score = analysis_data['color_score']
                    elif 'residual_score' in analysis_data:
                        score = analysis_data['residual_score']
                    elif 'temporal_score' in analysis_data:
                        score = analysis_data['temporal_score']
                    elif 'score' in analysis_data:
                        score = analysis_data['score']
                    
                    details = analysis_data.get('details', {})
                    if isinstance(details, dict):
                        suspicious = details.get('suspicious', False)
                    else:
                        suspicious = False
                    
                    scores.append(score)
                    all_suspicious.append(suspicious)
            
            if scores:
                avg_score = np.mean(scores)
                max_score = np.max(scores)
                suspicious_ratio = np.mean(all_suspicious)
                
                aggregated[analysis_type] = {
                    'score': float(avg_score),
                    'details': {
                        'avg_score': float(avg_score),
                        'max_score': float(max_score),
                        'suspicious_ratio': float(suspicious_ratio),
                        'suspicious': bool(suspicious_ratio > 0.3),
                        'frames_analyzed': int(len(scores))
                    }
                }
        
        return aggregated
    
    def analyze_video(self, video_path, max_frames=30):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': 'Não foi possível abrir o vídeo'}
        
        frame_analyses = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            analysis = self.analyze_frame(frame)
            frame_analyses.append(analysis)
            frame_count += 1
            
            print(f"Analisando frame {frame_count}/{max_frames}...")
        
        cap.release()
        
        if frame_analyses:
            aggregated_analysis = self.aggregate_frame_analyses(frame_analyses)
            report = self.generate_report(aggregated_analysis)
            return report
        else:
            return {'error': 'Nenhum frame foi analisado com sucesso'}

def main():
    print("="*60)
    print("🔍 DETECTOR DE DEEPFAKES - VERSÃO CORRIGIDA")
    print("="*60)
    
    detector = DeepfakeDetector()
    
    while True:
        print("\n📋 OPÇÕES:")
        print("1 - Analisar vídeo/GIF")
        print("2 - Webcam em tempo real")
        print("3 - Analisar imagem")
        print("4 - Configurar thresholds")
        print("0 - Sair")
        
        choice = input("\nEscolha: ").strip()
        
        if choice == "1":
            video_path = input("Caminho do vídeo/GIF: ").strip()
            if not video_path or not os.path.exists(video_path):
                print("❌ Arquivo não encontrado!")
                continue
            
            max_frames = input("Max frames (padrão 30): ").strip()
            try:
                max_frames = int(max_frames) if max_frames else 30
            except:
                max_frames = 30
            
            try:
                report = detector.analyze_video(video_path, max_frames)
                
                if 'error' in report:
                    print(f"❌ {report['error']}")
                else:
                    print(f"\n{'='*60}")
                    print(f"📋 RELATÓRIO DETALHADO DE ANÁLISE")
                    print(f"{'='*60}")
                    print(f"🎯 AVALIAÇÃO: {report['overall_assessment']}")
                    print(f"📊 CONFIANÇA: {report['confidence_level']:.2f}")
                    print(f"📁 ARQUIVO: {os.path.basename(video_path)}")
                    
                    print(f"\n{'='*40}")
                    print(f"🔍 INDICADORES ANALISADOS:")
                    print(f"{'='*40}")
                    
                    friendly_names = {
                        'edge_analysis': '🔲 Análise de Bordas',
                        'spectral_analysis': '📊 Análise Espectral (FFT/DCT)',
                        'color_analysis': '🎨 Consistência de Cores',
                        'residual_analysis': '🔍 Filtro Mediano (Halos)',
                        'temporal_analysis': '⏱️ Estabilidade Temporal'
                    }
                    
                    for analysis_name, result in report['detailed_analysis'].items():
                        score = result['score']
                        status = result['status']
                        name = friendly_names.get(analysis_name, analysis_name)
                        
                        if status == 'SUSPEITO':
                            print(f"🚨 {name}")
                            print(f"   Status: {status} | Score: {score:.3f}")
                        else:
                            print(f"✅ {name}")
                            print(f"   Status: {status} | Score: {score:.3f}")
                    
                    print(f"\n{'='*40}")
                    print(f"💡 RECOMENDAÇÕES:")
                    print(f"{'='*40}")
                    for i, rec in enumerate(report['recommendations'], 1):
                        print(f"{i}. {rec}")
                    
                    filename = f"relatorio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
                    print(f"\n💾 Relatório completo salvo: {filename}")
                    
            except Exception as e:
                print(f"❌ Erro: {e}")
        
        elif choice == "2":
            print("\n📹 Webcam ativa...")
            print("Q=Sair | C=Capturar | S=Pausar")
            
            try:
                cap = cv2.VideoCapture(0)
                frame_count = 0
                analyzing = True
                last_analysis = None
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    display_frame = frame.copy()
                    
                    if frame_count % 20 == 0 and analyzing:
                        analysis = detector.analyze_frame(frame)
                        last_analysis = analysis
                        
                        if analysis:
                            edge_score = analysis.get('edge_analysis', {}).get('edge_score', 0)
                            spectral_score = analysis.get('spectral_analysis', {}).get('spectral_score', 0)
                            color_score = analysis.get('color_analysis', {}).get('color_score', 0)
                            temporal_score = analysis.get('temporal_analysis', {}).get('temporal_score', 0)
                            
                            scores = [edge_score, spectral_score, color_score, temporal_score]
                            avg_score = np.mean([s for s in scores if s > 0])
                            
                            if avg_score > 0.2:
                                color = (0, 0, 255)
                                status = "SUSPEITO"
                            elif avg_score > 0.1:
                                color = (0, 165, 255)
                                status = "MODERADO"
                            else:
                                color = (0, 255, 0)
                                status = "NORMAL"
                    
                    roi, roi_coords = detector.detect_roi(display_frame)
                    if roi_coords:
                        x, y, w, h = roi_coords
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), color if last_analysis else (0, 255, 0), 3)
                        cv2.putText(display_frame, f"FACE: {status if last_analysis else 'DETECTADA'}", 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color if last_analysis else (0, 255, 0), 2)
                    
                    if last_analysis:
                        y_pos = 30
                        cv2.putText(display_frame, f'=== ANALISE DEEPFAKE ===', (10, y_pos), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        y_pos += 30
                        cv2.putText(display_frame, f'Score: {avg_score:.3f}', (10, y_pos), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        y_pos += 25
                        cv2.putText(display_frame, f'Status: {status}', (10, y_pos), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    cv2.imshow('Detector Deepfake', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('c') and last_analysis:
                        print(f"\n📸 ANÁLISE CAPTURADA:")
                        print(f"   Score: {avg_score:.3f}")
                        print(f"   Status: {status}")
                    elif key == ord('s'):
                        analyzing = not analyzing
                        print(f"{'PAUSADO' if not analyzing else 'ATIVO'}")
                    
                    frame_count += 1
                
                cap.release()
                cv2.destroyAllWindows()
                
            except Exception as e:
                print(f"❌ Erro: {e}")
        
        elif choice == "3":
            image_path = input("Caminho da imagem: ").strip()
            if not image_path or not os.path.exists(image_path):
                print("❌ Arquivo não encontrado")
                continue
            
            try:
                frame = cv2.imread(image_path)
                if frame is None:
                    print("❌ Não foi possível carregar imagem")
                    continue
                
                print(f"\n🔄 Analisando imagem: {os.path.basename(image_path)}")
                
                analysis = detector.analyze_frame(frame)
                
                # Monta análises para relatório
                analyses = {}
                for analysis_type in ['edge_analysis', 'spectral_analysis', 'color_analysis', 'residual_analysis']:
                    if analysis_type in analysis and isinstance(analysis[analysis_type], dict):
                        analysis_data = analysis[analysis_type]
                        
                        score = 0
                        if 'edge_score' in analysis_data:
                            score = analysis_data['edge_score']
                        elif 'spectral_score' in analysis_data:
                            score = analysis_data['spectral_score']  
                        elif 'color_score' in analysis_data:
                            score = analysis_data['color_score']
                        elif 'residual_score' in analysis_data:
                            score = analysis_data['residual_score']
                        
                        analyses[analysis_type] = {
                            'score': float(score),
                            'details': analysis_data.get('details', {})
                        }
                
                report = detector.generate_report(analyses)
                
                print(f"\n{'='*60}")
                print(f"📋 RELATÓRIO DE ANÁLISE - IMAGEM")
                print(f"{'='*60}")
                print(f"🎯 AVALIAÇÃO: {report['overall_assessment']}")
                print(f"📊 CONFIANÇA: {report['confidence_level']:.2f}")
                print(f"📁 ARQUIVO: {os.path.basename(image_path)}")
                
                print(f"\n{'='*40}")
                print(f"🔍 INDICADORES ANALISADOS:")
                print(f"{'='*40}")
                
                friendly_names = {
                    'edge_analysis': '🔲 Análise de Bordas',
                    'spectral_analysis': '📊 Análise Espectral (FFT/DCT)',
                    'color_analysis': '🎨 Consistência de Cores',
                    'residual_analysis': '🔍 Filtro Mediano (Halos)'
                }
                
                for analysis_name, result in report['detailed_analysis'].items():
                    score = result['score']
                    status = result['status']
                    name = friendly_names.get(analysis_name, analysis_name)
                    
                    if status == 'SUSPEITO':
                        print(f"🚨 {name}")
                        print(f"   Status: {status} | Score: {score:.3f}")
                    else:
                        print(f"✅ {name}")
                        print(f"   Status: {status} | Score: {score:.3f}")
                
                print(f"\n{'='*40}")
                print(f"💡 RECOMENDAÇÕES:")
                print(f"{'='*40}")
                for i, rec in enumerate(report['recommendations'], 1):
                    print(f"{i}. {rec}")
                
                filename = f"relatorio_imagem_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
                print(f"\n💾 Relatório completo salvo: {filename}")
                
            except Exception as e:
                print(f"❌ Erro: {e}")
        
        elif choice == "4":
            print("\n⚙️ THRESHOLDS ATUAIS (CORRIGIDOS):")
            for key, value in detector.thresholds.items():
                print(f"   {key}: {value}")
            
            print("\nNovos valores (Enter = manter):")
            for key in detector.thresholds.keys():
                current = detector.thresholds[key]
                new_value = input(f"{key} (atual: {current}): ").strip()
                if new_value:
                    try:
                        detector.thresholds[key] = float(new_value)
                        print(f"✅ {key} = {new_value}")
                    except:
                        print(f"❌ Valor inválido")
        
        elif choice == "0":
            print("👋 Encerrando...")
            break
        
        else:
            print("❌ Opção inválida")

if __name__ == "__main__":
    main()
