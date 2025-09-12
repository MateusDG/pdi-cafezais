import cv2
import numpy as np
import pickle
import os
from typing import List, Tuple, Dict, Any, Optional, Union
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib

from .ml_features import FeatureExtractor


class ClassicalMLWeedDetector:
    """
    Sistema de detecção de ervas daninhas usando algoritmos clássicos de Machine Learning.
    
    Suporta:
    - SVM (Support Vector Machine)
    - Random Forest
    - k-NN (k-Nearest Neighbors)
    - Naive Bayes
    
    Características extraídas:
    - Cor (RGB, HSV, índices de vegetação)
    - Textura (GLCM, LBP)
    - Forma e geometria (momentos, circularidade, etc.)
    """
    
    def __init__(self, model_dir: str = "models/classical_ml"):
        self.model_dir = model_dir
        self.feature_extractor = FeatureExtractor()
        
        # Modelos disponíveis
        self.models = {
            'svm': None,
            'random_forest': None,
            'knn': None,
            'naive_bayes': None
        }
        
        # Scalers para cada modelo
        self.scalers = {
            'svm': StandardScaler(),
            'random_forest': StandardScaler(),  
            'knn': StandardScaler(),
            'naive_bayes': StandardScaler()
        }
        
        # Label encoder
        self.label_encoder = LabelEncoder()
        
        # Criar diretório de modelos se não existir
        os.makedirs(model_dir, exist_ok=True)
        
    def prepare_training_data(self, images_and_labels: List[Tuple[np.ndarray, np.ndarray, str]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara dados de treinamento extraindo características das imagens.
        
        Args:
            images_and_labels: Lista de tuplas (imagem_rgb, máscara_binária, classe)
                             classe pode ser 'weed' ou 'coffee' ou 'soil'
            
        Returns:
            Tupla (características, labels) prontas para treinamento
        """
        all_features = []
        all_labels = []
        
        print(f"Processando {len(images_and_labels)} amostras de treinamento...")
        
        for i, (img, mask, label) in enumerate(images_and_labels):
            if i % 10 == 0:
                print(f"Processando amostra {i+1}/{len(images_and_labels)}")
            
            # Extrair características da região
            features = self.feature_extractor.extract_region_features(img, mask)
            
            # Converter para lista ordenada
            feature_vector = self._features_dict_to_vector(features)
            
            all_features.append(feature_vector)
            all_labels.append(label)
        
        # Converter para numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"Dataset preparado: {X.shape[0]} amostras, {X.shape[1]} características")
        
        return X, y
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray, 
                        test_size: float = 0.2, random_state: int = 42) -> Dict[str, Dict[str, Any]]:
        """
        Treina todos os modelos clássicos disponíveis.
        
        Args:
            X: Características das amostras
            y: Labels das amostras  
            test_size: Proporção para teste
            random_state: Semente aleatória
            
        Returns:
            Dicionário com resultados de cada modelo
        """
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Encoder para labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        results = {}
        
        # 1. SVM
        print("Treinando SVM...")
        results['svm'] = self._train_svm(X_train, X_test, y_train_encoded, y_test_encoded)
        
        # 2. Random Forest
        print("Treinando Random Forest...")
        results['random_forest'] = self._train_random_forest(X_train, X_test, y_train_encoded, y_test_encoded)
        
        # 3. k-NN
        print("Treinando k-NN...")
        results['knn'] = self._train_knn(X_train, X_test, y_train_encoded, y_test_encoded)
        
        # 4. Naive Bayes
        print("Treinando Naive Bayes...")
        results['naive_bayes'] = self._train_naive_bayes(X_train, X_test, y_train_encoded, y_test_encoded)
        
        return results
    
    def _train_svm(self, X_train: np.ndarray, X_test: np.ndarray, 
                   y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Treina modelo SVM com busca em grid."""
        
        # Escalar dados
        X_train_scaled = self.scalers['svm'].fit_transform(X_train)
        X_test_scaled = self.scalers['svm'].transform(X_test)
        
        # Grid search para hiperparâmetros
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'poly']
        }
        
        svm_model = SVC(random_state=42, probability=True)
        grid_search = GridSearchCV(svm_model, param_grid, cv=5, 
                                 scoring='accuracy', n_jobs=-1, verbose=1)
        
        # Treinar
        grid_search.fit(X_train_scaled, y_train)
        
        # Melhor modelo
        best_svm = grid_search.best_estimator_
        self.models['svm'] = best_svm
        
        # Avaliar
        y_pred = best_svm.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(best_svm, X_train_scaled, y_train, cv=5)
        
        return {
            'model': best_svm,
            'best_params': grid_search.best_params_,
            'accuracy': accuracy,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred, 
                                                         target_names=self.label_encoder.classes_),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def _train_random_forest(self, X_train: np.ndarray, X_test: np.ndarray,
                           y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Treina modelo Random Forest."""
        
        # Escalar dados (Random Forest não precisa, mas para consistência)
        X_train_scaled = self.scalers['random_forest'].fit_transform(X_train)
        X_test_scaled = self.scalers['random_forest'].transform(X_test)
        
        # Grid search
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf_model, param_grid, cv=5,
                                 scoring='accuracy', n_jobs=-1, verbose=1)
        
        # Treinar
        grid_search.fit(X_train_scaled, y_train)
        
        # Melhor modelo
        best_rf = grid_search.best_estimator_
        self.models['random_forest'] = best_rf
        
        # Avaliar
        y_pred = best_rf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(best_rf, X_train_scaled, y_train, cv=5)
        
        # Feature importance
        feature_names = self.feature_extractor.get_all_feature_names()
        feature_importance = dict(zip(feature_names, best_rf.feature_importances_))
        
        return {
            'model': best_rf,
            'best_params': grid_search.best_params_,
            'accuracy': accuracy,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'classification_report': classification_report(y_test, y_pred,
                                                         target_names=self.label_encoder.classes_),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def _train_knn(self, X_train: np.ndarray, X_test: np.ndarray,
                   y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Treina modelo k-NN."""
        
        # Escalar dados (importante para k-NN)
        X_train_scaled = self.scalers['knn'].fit_transform(X_train)
        X_test_scaled = self.scalers['knn'].transform(X_test)
        
        # Grid search
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
        
        knn_model = KNeighborsClassifier()
        grid_search = GridSearchCV(knn_model, param_grid, cv=5,
                                 scoring='accuracy', n_jobs=-1, verbose=1)
        
        # Treinar
        grid_search.fit(X_train_scaled, y_train)
        
        # Melhor modelo
        best_knn = grid_search.best_estimator_
        self.models['knn'] = best_knn
        
        # Avaliar
        y_pred = best_knn.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(best_knn, X_train_scaled, y_train, cv=5)
        
        return {
            'model': best_knn,
            'best_params': grid_search.best_params_,
            'accuracy': accuracy,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred,
                                                         target_names=self.label_encoder.classes_),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def _train_naive_bayes(self, X_train: np.ndarray, X_test: np.ndarray,
                          y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Treina modelo Naive Bayes."""
        
        # Escalar dados
        X_train_scaled = self.scalers['naive_bayes'].fit_transform(X_train)
        X_test_scaled = self.scalers['naive_bayes'].transform(X_test)
        
        # Naive Bayes não tem muitos hiperparâmetros
        nb_model = GaussianNB()
        
        # Treinar
        nb_model.fit(X_train_scaled, y_train)
        self.models['naive_bayes'] = nb_model
        
        # Avaliar
        y_pred = nb_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(nb_model, X_train_scaled, y_train, cv=5)
        
        return {
            'model': nb_model,
            'accuracy': accuracy,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred,
                                                         target_names=self.label_encoder.classes_),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def predict_regions(self, img: np.ndarray, regions: List[Tuple[np.ndarray, np.ndarray]], 
                       model_name: str = 'random_forest') -> List[Dict[str, Any]]:
        """
        Classifica regiões da imagem usando modelo treinado.
        
        Args:
            img: Imagem RGB
            regions: Lista de tuplas (contorno, máscara_binária)
            model_name: Nome do modelo a usar
            
        Returns:
            Lista de predições para cada região
        """
        if model_name not in self.models or self.models[model_name] is None:
            raise ValueError(f"Modelo {model_name} não foi treinado")
        
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        
        predictions = []
        
        for i, (contour, mask) in enumerate(regions):
            # Extrair características
            features = self.feature_extractor.extract_region_features(img, mask)
            feature_vector = self._features_dict_to_vector(features)
            
            # Normalizar
            feature_vector_scaled = scaler.transform([feature_vector])
            
            # Predizer
            prediction = model.predict(feature_vector_scaled)[0]
            prediction_proba = model.predict_proba(feature_vector_scaled)[0]
            
            # Decodificar label
            predicted_class = self.label_encoder.inverse_transform([prediction])[0]
            
            # Probabilidades por classe
            class_probabilities = dict(zip(self.label_encoder.classes_, prediction_proba))
            
            predictions.append({
                'region_id': i,
                'predicted_class': predicted_class,
                'confidence': float(max(prediction_proba)),
                'class_probabilities': {k: float(v) for k, v in class_probabilities.items()},
                'contour': contour,
                'features': features
            })
        
        return predictions
    
    def detect_weeds_ml(self, img: np.ndarray, model_name: str = 'random_forest',
                       confidence_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Detecta ervas daninhas usando modelo de ML clássico.
        
        Args:
            img: Imagem RGB
            model_name: Nome do modelo ('svm', 'random_forest', 'knn', 'naive_bayes')
            confidence_threshold: Threshold mínimo de confiança
            
        Returns:
            Resultado da detecção com estatísticas
        """
        # Primeiro usar segmentação tradicional para encontrar regiões candidatas
        from app.services.processing.weed import detect_weeds_robust
        
        # Detectar regiões vegetativas
        segmentation_result = detect_weeds_robust(img)
        contours = segmentation_result['contours']
        
        if not contours:
            return self._create_empty_ml_result(img, model_name)
        
        # Criar máscaras para cada região
        regions = []
        height, width = img.shape[:2]
        
        for contour in contours:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            regions.append((contour, mask))
        
        # Classificar regiões
        predictions = self.predict_regions(img, regions, model_name)
        
        # Filtrar ervas daninhas com confiança suficiente
        weed_contours = []
        weed_confidences = []
        coffee_contours = []
        
        for pred in predictions:
            if (pred['predicted_class'] == 'weed' and 
                pred['confidence'] >= confidence_threshold):
                weed_contours.append(pred['contour'])
                weed_confidences.append(pred['confidence'])
            elif pred['predicted_class'] == 'coffee':
                coffee_contours.append(pred['contour'])
        
        # Calcular estatísticas
        total_weed_area = sum(cv2.contourArea(c) for c in weed_contours)
        image_area = height * width
        weed_percentage = (total_weed_area / image_area) * 100
        
        # Criar imagem anotada
        annotated = self._annotate_ml_results(img, predictions, confidence_threshold)
        
        return {
            'annotated_image': annotated,
            'contours': weed_contours,
            'weed_count': len(weed_contours),
            'total_weed_area': int(total_weed_area),
            'weed_percentage': round(weed_percentage, 2),
            'image_area': image_area,
            'algorithm': f'Classical ML - {model_name.upper()}',
            'model_used': model_name,
            'confidence_threshold': confidence_threshold,
            'predictions': predictions,
            'avg_confidence': round(np.mean(weed_confidences), 3) if weed_confidences else 0.0,
            'coffee_regions_found': len(coffee_contours)
        }
    
    def save_models(self):
        """Salva todos os modelos treinados."""
        for model_name, model in self.models.items():
            if model is not None:
                model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
                scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
                
                joblib.dump(model, model_path)
                joblib.dump(self.scalers[model_name], scaler_path)
        
        # Salvar label encoder
        encoder_path = os.path.join(self.model_dir, "label_encoder.pkl")
        joblib.dump(self.label_encoder, encoder_path)
        
        print(f"Modelos salvos em {self.model_dir}")
    
    def load_models(self):
        """Carrega modelos salvos."""
        for model_name in self.models.keys():
            model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
            scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.models[model_name] = joblib.load(model_path)
                self.scalers[model_name] = joblib.load(scaler_path)
                print(f"Modelo {model_name} carregado")
        
        # Carregar label encoder
        encoder_path = os.path.join(self.model_dir, "label_encoder.pkl")
        if os.path.exists(encoder_path):
            self.label_encoder = joblib.load(encoder_path)
            print("Label encoder carregado")
    
    def _features_dict_to_vector(self, features_dict: Dict[str, float]) -> List[float]:
        """Converte dicionário de características em vetor ordenado."""
        feature_names = self.feature_extractor.get_all_feature_names()
        return [features_dict.get(name, 0.0) for name in feature_names]
    
    def _create_empty_ml_result(self, img: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Cria resultado vazio quando não há regiões detectadas."""
        return {
            'annotated_image': img.copy(),
            'contours': [],
            'weed_count': 0,
            'total_weed_area': 0,
            'weed_percentage': 0.0,
            'image_area': img.shape[0] * img.shape[1],
            'algorithm': f'Classical ML - {model_name.upper()}',
            'model_used': model_name,
            'predictions': [],
            'avg_confidence': 0.0,
            'coffee_regions_found': 0
        }
    
    def _annotate_ml_results(self, img: np.ndarray, predictions: List[Dict[str, Any]], 
                           confidence_threshold: float) -> np.ndarray:
        """Anota imagem com resultados da classificação ML."""
        annotated = img.copy()
        
        # Cores por classe
        colors = {
            'weed': (255, 0, 0),      # Vermelho
            'coffee': (0, 255, 0),    # Verde
            'soil': (139, 69, 19),    # Marrom
            'unknown': (128, 128, 128) # Cinza
        }
        
        weed_count = 0
        coffee_count = 0
        
        for pred in predictions:
            contour = pred['contour']
            predicted_class = pred['predicted_class']
            confidence = pred['confidence']
            
            # Cor baseada na classe e confiança
            if confidence >= confidence_threshold:
                color = colors.get(predicted_class, colors['unknown'])
                thickness = 3
            else:
                color = colors['unknown']
                thickness = 1
            
            # Desenhar contorno
            cv2.drawContours(annotated, [contour], -1, color, thickness)
            
            # Contar por classe
            if predicted_class == 'weed' and confidence >= confidence_threshold:
                weed_count += 1
            elif predicted_class == 'coffee':
                coffee_count += 1
            
            # Adicionar texto com confiança
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                text = f"{predicted_class[:4]}: {confidence:.2f}"
                
                # Background para o texto
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(annotated, 
                            (cx - text_size[0]//2 - 2, cy - text_size[1] - 2),
                            (cx + text_size[0]//2 + 2, cy + 2), 
                            (255, 255, 255), -1)
                
                cv2.putText(annotated, text, (cx - text_size[0]//2, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Info box
        info_height = 100
        cv2.rectangle(annotated, (10, 10), (450, info_height), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, 10), (450, info_height), (255, 255, 255), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(annotated, f"ML Classification Results", 
                   (15, 30), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Weeds: {weed_count} | Coffee: {coffee_count}", 
                   (15, 50), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Confidence threshold: {confidence_threshold:.2f}", 
                   (15, 70), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Red=Weed, Green=Coffee, Brown=Soil, Gray=Low confidence", 
                   (15, 90), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        
        return annotated