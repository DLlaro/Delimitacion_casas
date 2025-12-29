import tensorflow as tf

def masked_sparse_cce(y_true, y_pred):
    """
    Sparse categorical crossentropy que ignora píxeles con label 255
    """
    y_true = tf.squeeze(y_true, axis=-1)  # (B,H,W)
    y_true = tf.cast(y_true, tf.int32)
    # Crear máscara para píxeles válidos
    mask = tf.not_equal(y_true, 255)
    mask = tf.cast(mask, tf.float32)
    
    # Reemplazar 255 con 0 para evitar errores
    y_true_safe = tf.where(mask > 0, y_true, tf.zeros_like(y_true))
    
    # Calcular loss por píxel
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true_safe, y_pred, from_logits=False
    )
    
    # Aplicar máscara y promediar solo sobre píxeles válidos
    loss = loss * mask
    
    # Evitar división por cero
    num_valid = tf.maximum(tf.reduce_sum(mask), 1.0)
    
    return tf.reduce_sum(loss) / num_valid


class MaskedIoUPerClass(tf.keras.metrics.Metric):
    def __init__(self, num_classes=3, class_names=None, name='masked_iou_per_class', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        
        # Matriz de confusión acumulativa
        self.total_cm = self.add_weight(
            name='total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer='zeros',
            dtype=tf.float32
        )
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        
        # Convertir a int32
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        
        # Máscara para ignorar 255
        mask = tf.not_equal(y_true, 255)
        
        # Aplicar máscara
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        
        # Calcular confusion matrix
        current_cm = tf.math.confusion_matrix(
            y_true,
            y_pred,
            num_classes=self.num_classes,
            dtype=tf.float32
        )
        
        self.total_cm.assign_add(current_cm)
    
    def result(self):
        """Retorna el mIoU promedio (para compatibilidad con callbacks)"""
        iou_per_class = self._compute_iou_per_class()
        return tf.reduce_mean(iou_per_class)
    
    def _compute_iou_per_class(self):
        """Calcula IoU para cada clase"""
        sum_over_row = tf.reduce_sum(self.total_cm, axis=0)  # Predicciones
        sum_over_col = tf.reduce_sum(self.total_cm, axis=1)  # Ground truth
        diag = tf.linalg.diag_part(self.total_cm)  # True positives
        
        # IoU = TP / (TP + FP + FN)
        denominator = sum_over_row + sum_over_col - diag
        
        iou = tf.where(
            denominator > 0,
            diag / denominator,
            0.0
        )
        
        return iou
    
    def get_iou_per_class(self):
        """Método para obtener IoU por clase (usar después del entrenamiento)"""
        iou = self._compute_iou_per_class()
        return {self.class_names[i]: float(iou[i].numpy()) for i in range(self.num_classes)}
    
    def reset_state(self):
        self.total_cm.assign(tf.zeros_like(self.total_cm))


class MaskedSparseCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, num_classes=3, class_names=None, name='masked_acc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        
        # Acumuladores globales
        self.total_correct = self.add_weight(name='total_correct', initializer='zeros', dtype=tf.float32)
        self.total_pixels = self.add_weight(name='total_pixels', initializer='zeros', dtype=tf.float32)
        
        # Acumuladores por clase
        self.correct_per_class = self.add_weight(
            name='correct_per_class',
            shape=(num_classes,),
            initializer='zeros',
            dtype=tf.float32
        )
        self.pixels_per_class = self.add_weight(
            name='pixels_per_class',
            shape=(num_classes,),
            initializer='zeros',
            dtype=tf.float32
        )
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        
        # Convertir a int32
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        
        # Máscara para ignorar 255
        mask = tf.not_equal(y_true, 255)
        
        # Aplicar máscara
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        
        # Accuracy global
        correct = tf.cast(tf.equal(y_true_masked, y_pred_masked), tf.float32)
        self.total_correct.assign_add(tf.reduce_sum(correct))
        self.total_pixels.assign_add(tf.cast(tf.size(y_true_masked), tf.float32))
        
        # Accuracy por clase
        for class_id in range(self.num_classes):
            # Máscara para la clase actual
            class_mask = tf.equal(y_true_masked, class_id)
            
            if tf.reduce_any(class_mask):
                y_true_class = tf.boolean_mask(y_true_masked, class_mask)
                y_pred_class = tf.boolean_mask(y_pred_masked, class_mask)
                
                correct_class = tf.cast(tf.equal(y_true_class, y_pred_class), tf.float32)
                
                # Actualizar acumuladores
                self.correct_per_class.scatter_add(
                    tf.IndexedSlices(tf.reduce_sum(correct_class), [class_id])
                )
                self.pixels_per_class.scatter_add(
                    tf.IndexedSlices(tf.cast(tf.size(y_true_class), tf.float32), [class_id])
                )
    
    def result(self):
        """Retorna accuracy global (para compatibilidad con callbacks)"""
        return self.total_correct / tf.maximum(self.total_pixels, 1.0)
    
    def get_accuracy_per_class(self):
        """Obtiene accuracy por clase"""
        acc_per_class = tf.where(
            self.pixels_per_class > 0,
            self.correct_per_class / self.pixels_per_class,
            0.0
        )
        return {self.class_names[i]: float(acc_per_class[i].numpy()) for i in range(self.num_classes)}
    
    def reset_state(self):
        self.total_correct.assign(0.0)
        self.total_pixels.assign(0.0)
        self.correct_per_class.assign(tf.zeros((self.num_classes,)))
        self.pixels_per_class.assign(tf.zeros((self.num_classes,)))

class PrintMetricsPerClass(tf.keras.callbacks.Callback):
    def __init__(self, metric_names=['acc', 'miou']):
        super().__init__()
        self.metric_names = metric_names
    
    def on_epoch_end(self, epoch, logs=None):
        print(f"\n📊 Época {epoch+1} - Métricas por clase:")
        
        for metric in self.model.metrics:
            if metric.name in self.metric_names:
                if hasattr(metric, 'get_accuracy_per_class'):
                    results = metric.get_accuracy_per_class()
                    print(f"\n  Accuracy por clase:")
                    for class_name, value in results.items():
                        print(f"    {class_name}: {value:.4f}")
                
                elif hasattr(metric, 'get_iou_per_class'):
                    results = metric.get_iou_per_class()
                    print(f"\n  IoU por clase:")
                    for class_name, value in results.items():
                        print(f"    {class_name}: {value:.4f}")