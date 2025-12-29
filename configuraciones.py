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


class MeanIoUPerClass(tf.keras.metrics.Metric):
    """
    Métrica de Mean Intersection over Union que:
    - Ignora píxeles con label 255
    - Calcula IoU por clase y promedio
    - Mantiene matriz de confusión acumulativa
    """
    def __init__(self, num_classes, class_names=None, name='miou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.class_names = class_names or [f"Clase_{i}" for i in range(num_classes)]
        
        # Matriz de confusión: confusion_matrix[true_class, pred_class]
        self.confusion_matrix = self.add_weight(
            name='confusion_matrix',
            shape=(num_classes, num_classes),
            initializer='zeros',
            dtype=tf.float32
        )
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        
        # Máscara para ignorar 255
        mask = tf.not_equal(y_true, 255)
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        
        # Actualizar matriz de confusión usando índices planos
        indices = y_true_masked * self.num_classes + y_pred_masked
        
        updates = tf.ones_like(indices, dtype=tf.float32)
        flat_confusion = tf.math.unsorted_segment_sum(
            updates,
            indices,
            num_segments=self.num_classes * self.num_classes
        )
        
        confusion_batch = tf.reshape(flat_confusion, (self.num_classes, self.num_classes))
        self.confusion_matrix.assign_add(confusion_batch)
    
    def result(self):
        """
        Retorna Mean IoU promediado solo sobre clases presentes en el dataset
        """
        tp = tf.linalg.diag_part(self.confusion_matrix)
        fp = tf.reduce_sum(self.confusion_matrix, axis=0) - tp
        fn = tf.reduce_sum(self.confusion_matrix, axis=1) - tp
        
        iou = tp / (tp + fp + fn + 1e-7)
        
        # Promediar solo sobre clases que tienen píxeles en ground truth
        valid_classes = tf.cast(tp + fn > 0, tf.float32)
        
        # Sumar IoUs de clases válidas y dividir por número de clases válidas
        sum_valid_iou = tf.reduce_sum(iou * valid_classes)
        num_valid_classes = tf.maximum(tf.reduce_sum(valid_classes), 1.0)
        
        return sum_valid_iou / num_valid_classes
    
    def get_iou_per_class(self):
        """Retorna diccionario con IoU por clase"""
        tp = tf.linalg.diag_part(self.confusion_matrix)
        fp = tf.reduce_sum(self.confusion_matrix, axis=0) - tp
        fn = tf.reduce_sum(self.confusion_matrix, axis=1) - tp
        
        iou = tp / (tp + fp + fn + 1e-7)
        iou_numpy = iou.numpy()
        
        return {
            self.class_names[i]: float(iou_numpy[i])
            for i in range(self.num_classes)
        }
    
    def reset_state(self):
        self.confusion_matrix.assign(tf.zeros((self.num_classes, self.num_classes)))


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
        
        # Accuracy por clase (vectorizado)
        correct_indicator = tf.cast(tf.equal(y_true_masked, y_pred_masked), tf.float32)
        
        # Sumar correctos por clase usando unsorted_segment_sum
        correct_per_class_batch = tf.math.unsorted_segment_sum(
            correct_indicator,
            y_true_masked,
            num_segments=self.num_classes
        )
        
        # Contar píxeles por clase
        ones = tf.ones_like(y_true_masked, dtype=tf.float32)
        pixels_per_class_batch = tf.math.unsorted_segment_sum(
            ones,
            y_true_masked,
            num_segments=self.num_classes
        )
        
        # Actualizar variables
        self.correct_per_class.assign_add(correct_per_class_batch)
        self.pixels_per_class.assign_add(pixels_per_class_batch)
    
    def result(self):
        """Retorna accuracy global (para Keras)"""
        return self.total_correct / (self.total_pixels + 1e-7)
    
    def get_accuracy_per_class(self):
        """Retorna diccionario con accuracy por clase (para callback)"""
        class_acc = self.correct_per_class / (self.pixels_per_class + 1e-7)
        return {
            self.class_names[i]: float(class_acc[i].numpy())
            for i in range(self.num_classes)
        }
    
    def reset_state(self):
        """Reinicia todos los acumuladores"""
        self.total_correct.assign(0.0)
        self.total_pixels.assign(0.0)
        self.correct_per_class.assign(tf.zeros((self.num_classes,), dtype=tf.float32))
        self.pixels_per_class.assign(tf.zeros((self.num_classes,), dtype=tf.float32))

class PrintMetricsPerClass(tf.keras.callbacks.Callback):
    def __init__(self, metric_names=['acc', 'miou']):
        super().__init__()
        self.metric_names = metric_names
    
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nÉpoca {epoch+1} - Métricas por clase:")
        
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