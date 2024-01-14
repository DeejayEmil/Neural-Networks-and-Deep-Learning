# Especificamos los valores de los hiperparametros con la funcion flags() del paquete tfruns

# flag_integer(name, default, description = NULL)

FLAGS <- flags(flag_integer("dense_units", 50)) # 34 es el valor por defecto


##################
## Red neuronal ## 
##################

model <- keras_model_sequential() %>% 
  layer_dropout(0.2, input_shape = ncol(x_train)) %>% 
  layer_dense(units = FLAGS$dense_units, activation = "relu",
              kernel_constraint = constraint_maxnorm(3)) %>% 
  layer_dense(units = 1, activation = 'sigmoid')

# Compilacion

optimizador <- "sgd"

model %>% compile(loss = "binary_crossentropy",
                  optimizer = optimizador,   
                  metric = c("accuracy")) 
# Entrenamiento

history <- model %>% fit(x_train, y_train,
                         epochs = 50,
                         batch_size = 16,
                         validation_split = .2,
                         callbacks = callback_early_stopping(patience = 10, 
                                                             monitor = 'val_accuracy', 
                                                             restore_best_weights = TRUE))
