model.evaluate(val_data)
model.export(export_dir='.', tflite_filename='android.tflite')
