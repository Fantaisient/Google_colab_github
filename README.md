# Folder structure

main.py / main.ipynb – main script to start training initialises the models, initialises dataset, sets hyperparameters for training, and trains the model. No code implementation is required in these files. You are allowed to change hyperparameters. 

model.py – Discriminator and Generator classes initialisation. Tasks 2.10.1, 2.10.2, 2.10.3 to be implemented here. 

trainer.py – WGANTrainer class for training and additional utility functions for logging the losses and outputs. Tasks 2.10.4, 2.10.5, 2.10.6, 2.10.7 to be implemented here. 

check.py –  script to check the performance of the trained model on the fixed vector. 

fixed_20_z_for_check.pt – fixed vector for check.py.

S2_SCIPER.docx – template for tasks 2.10.8, 2.10.9. 


# Submission guidelines



Create a folder named S2_SCIPER, where the word “SCIPER” should be replaced with your
SCIPER ID. Please, follow the file structure and the naming convention as below.

<pre>

S2_SCIPER/
    S2_SCIPER.pdf – answers to tasks 2.10.8, 2.10.9 
    either main.py, or main.ipynb (not both, depends on what you have used to train your model)
    model.py
    trainer.py
    check.py
    fixed_20_z_for_check.pt
    report_images/
        fixed_z_for_check_ep0.jpg
        fixed_z_for_check_ep1.jpg
        ...
        fixed_z_for_check_epN.jpg 
        losses_ep0.jpg
        losses_ep1.jpg
        ...
        losses_epN.jpg
    model_weights/
        generator.pt, 
        discriminator.pt
</pre>

Folders report_images/ and model_weights/ and their content are generated during training. Please, make sure to add these files to your submission. Files generator.pt and discriminator.pt should contain the weights of the trained models and are updated after every epoch. 

Once you are ready for submission, submit your .zip folder to Moodle.
