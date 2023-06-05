# TF_filters_locations-2
🧸💬 Tensorflow for object locators, filters and convolution is enough for tracking objects from the plain background and we plan to add DCT ( discrete frequency transform for colours image ). This method is a filters method and different than our previous method detects an object by convolution and scaling or scanning the image for object similarity.

#### 🐑💬 The true skills of AI is problem transformation, this is parameters for snakes games. ####
```
coeff_01 = ball_y_value - player_y_value
coeff_02 = ball_x_value - player_x_value
coeff_03 = 5
coeff_04 = player_x_value - ball_x_value
coeff_05 = player_y_value - ball_y_value
```

#### 🐑💬 Filters - location on plain background ####
```
layer_1 = tf.keras.layers.Normalization(mean=3., variance=2.)( image_resized )
layer_2 = tf.keras.layers.Normalization(mean=4., variance=6.)( layer_1 )

image = tf.expand_dims( image_resized, axis=0, name="expand dimension" )
layer_3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')( image )

final_layer = tf.keras.layers.Conv2D(1, (2, 1), activation='relu')( layer_3 )
final_layer = tf.squeeze( final_layer, axis=0, name="squeeze" )
```

### Snake games ###
🥺💬 You can find projectiles of the objects and you can accelerate but you cannot deaccelerate, VST. 
![alt text](https://github.com/jkaewprateep/TF_filters_locations-2/blob/main/Snakes.gif)<br>

### Water World games ###
🧸💬 There are rules and speed to catch up, it is not only one way to accelerate speed by projection in time.
![alt text](https://github.com/jkaewprateep/TF_filters_locations-2/blob/main/waterworld.gif)

### Object location from TF_filters_locations ###
🐐💬 Its objective is to come close as fast as it active response, there is nowhere to locate 100% exact initial but continuing is more accuracy. Filters create multiple sizes of objects with different shapes and sizes but it is continuity located by the same acceleration.

![alt text](https://github.com/jkaewprateep/TF_filters_locations-2/blob/main/Simple%20Filter%20locations.gif)

## Files and directory ##
File name | Description |
--- | --- |
Simple Filter locations.gif | Object location on the plain screen using filters technique |
Snakes.gif | Application with object with dimension |
waterworld.gif | Application with objects |
README.md | Readme file |
