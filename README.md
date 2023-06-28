# Irish-title-representation

This repository documents my research on uncovering the connection between song titles and their musical features, such as melody, key, genre, rhythm, and more.

For more detailed explanations, you can refer to the following link: https://www.youtube.com/watch?v=fY0ocw9sbrg


## Summary
I am undertaking this project as part of a class at Sogang University with the guidance and assistance of my advisor, Professor Jeong (https://jdasam.github.io/maler/). I embarked on this project with the aim of exploring the correlations between song titles and their musical features. However, due to my limited experience and capacity to independently pursue this project in-depth, I had to alter my initial plan. Consequently, I utilized the representation vectors obtained from training the model to conditionally generate melodies for Irish folk music.

## Additional explanation
### Generation model
- Revision  
Initially, I attempted to develop my own generation model by referring to a research paper on VirtusoTune by Professor Jeong. Unfortunately, my model did not undergo adequate training for satisfactory generation. Due to time constraints, instead of debugging my model, I made the decision to modify and revise the VirtusoTune model itself to suit my specific purpose.

### Utilizing Representation Vectors from pretrained title embedding model
- Title Matching  
To address the limitation of title embedding alone in influencing melody generation in VirtusoTune, I generated 180 distinct sets of header embeddings using a pre-trained musical feature embedding model. These embeddings were then matched with a given title embedding obtained from a pre-trained title embedding model. By incorporating a sequential process of generating header information based on the given title embedding, these conditions exerted a greater influence compared to using header embedding alone.
Furthermore, the generation model was exclusively trained using the C key in various modes. I deliberately manipulated the output key information, which was generated based on the given title, to produce different key variations.

### generation script
<img width="939" alt="title_melody" src="https://github.com/clayryu/Irish-title-representation/assets/95623535/7f3b8875-500f-4776-80ce-20d6aa353f73">

### generated Irish folk music
<audio controls>
    <source src='model_measure_note_xl_seed_2_key_C minor.wav'>
</audio>
