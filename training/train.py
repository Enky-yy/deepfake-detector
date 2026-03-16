from models.deepfake_model import build_models

model = build_models()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy'
    metrics=['accuracy']
)

model.fit(
    train_df,
    validation_data=val_df,
    epochs=10)
)

model.save('trained_models/model.h5', overwrite=True)