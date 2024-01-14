# Especificamos los valores de los hiperparametros con la funcion flags() del paquete tfruns

FLAGS <- flags(
               flag_integer("dense_units", 50),
               flag_integer("epochs", 50),
               flag_string("optimizers", "sgd"),
               flag_integer("batch_size", 16))


##################
## Red neuronal ## 
##################

model <- keras_model_sequential() %>% 
  layer_dropout(.2, input_shape = ncol(x_train)) %>% 
  layer_dense(units = FLAGS$dense_units, activation = "relu",
              kernel_constraint = constraint_maxnorm(3)) %>% 
  layer_dense(units = 1, activation = 'sigmoid')

# Compilacion

model %>% compile(loss = "binary_crossentropy",
                  optimizer = FLAGS$optimizers,   
                  metric = c("accuracy"))

# Entrenamiento

history <- model %>% fit(x_train, y_train,
                         epochs = FLAGS$epochs,
                         batch_size = FLAGS$batch_size,
                         validation_split = 0.2)
