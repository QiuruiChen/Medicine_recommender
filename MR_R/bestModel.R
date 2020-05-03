rm(list=ls())
library(keras)
library(cloudml)
pair <- readRDS('data/pair.RDa')
pair<-pair[sample(nrow(pair)),]

n_drug <- length(unique(pair$WoundId))
n_adr <- length(unique(pair$Article))
k <-20

input_drug <- layer_input(shape = c(1),name="woundID")
input_adr <- layer_input(shape = c(1),name="articleID")

# embedding and flatten layers
embed_drug <- input_drug %>%
  layer_embedding(input_dim = n_drug, output_dim = k, input_length = 1) %>%
  layer_flatten()
embed_adr <- input_adr %>%
  layer_embedding(input_dim = n_adr, output_dim = k, input_length = 1) %>%
  layer_flatten()

pred <- layer_dot(list(embed_drug, embed_adr), axes = -1) %>%
  layer_dense(units = 1, activation = "sigmoid")

model <- keras_model(inputs = c(input_drug, input_adr), outputs = pred)
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('binary_accuracy')
)
summary(model)
# train the model
model %>% fit(
  x = list(
    matrix(pair$WoundId, ncol = 1),
    matrix(pair$Article, ncol = 1)
  ),
  y = matrix(pair$class, ncol = 1),
  class_weight = list("1" = 50.0, "0" = 1.0), # deal with unbalanced classes
  verbose = 1,
  callbacks =list(
    callback_early_stopping(monitor="val_loss", patience=40, verbose=1, mode=c('auto')),
    callback_reduce_lr_on_plateau(monitor = "val_loss",
                                  factor = 0.1,patience=50, verbose=1, mode=c('min'))
  ),
  epochs = 1000,
  batch_size = 4096,
  # validation_split = 0.2
  validation_data = list(list(
    matrix(pair$WoundId, ncol = 1),
    matrix(pair$Article, ncol = 1)
  ),matrix(pair$class, ncol = 1)
  )
)


export_savedmodel(model, "savedmodel")
cloudml_deploy("savedmodel", name = "collaborative_filter_rec",region="europe-west1",version = paste0("collaborative_filter_rec", "_3"))


cloudml_predict(list(list(as.vector(358),as.vector(19))), name="collaborative_filter_rec", 
                version = paste0("collaborative_filter_rec", "_3"),
                verbose = FALSE)

