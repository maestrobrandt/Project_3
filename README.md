# Music and Sensorimotor Coupling: What Music is Groovy?

![Project Image](https://media-exp1.licdn.com/dms/image/C4E12AQE0xkg-FrYSzg/article-cover_image-shrink_720_1280/0/1614708423020?e=1627516800&v=beta&t=sPLGeapkdNqmW3Tjqu2k8GMHejVFXJrJsDJrAhO_L2I)

>Discover what music is scientifically proven to make you move more

___

### Table of Contents
- [Description](#description)
- [Introduction](#introduction)
- [Story](#story)
- [Application](#application)
- [Future Consideration](#future-consideration)
- [Author Info](#author-info)

___

## Description

This project aimed to classify music as groovy or not groovy based on conclusions reached by a study at UC Davis on how certain music engages the brain's sensorimotor mechanisms better than others. It also included a web application that returns the "Groovy" songs of a given artist using a tuned Random-Forrest model.

#### Files
- The code for the project is in Music_and_Sensorimotor_Coupling_code.ipynb
- The app runs in streamlit, and will requre the Spotify_API.py and Spotify_App.py files

#### Technologies
- Python
- Jupyter Notebooks
- Classification Models
  - KNN
  - Logit
  - Naive Bayes
  - Decicion Tree
  - Random Forrest
  - Support Vector Machines
  - XGBoost
- Streamlit

___

## Introduction

Music, since the dawn of civilization, has been an integral part of the human experience, and continues to permeate through every culture on earth. It is used for a multitude of purposes such as worship, preparing for war, healing the sick, or announcing the arrival of royalty. Of course, the most obvious use for music in the present day is entertainment. In terms of self-entertainment, one can spend hours listening to their favorite band, exercising to an energetic playlist, or perhaps consoling themself with sad music after a tough break-up. In terms of public entertainment, there are concerts, background music in restaurants or stores, and dance music at parties or clubs. Hence, while the uses for music at the individual level are many, the applications to the commercial realm should not be ignored. It is a fact that music affects our emotions, so business owners would be smart to select music that keeps their customers happy. Similarly, music literally has the power to make us want to move and dance, and club owners and DJs seek to select the best music to make use of this phenomenon.

The reason that music has such a profound effect on our emotions is attributable in no small part to the activation of neurological mechanisms in response to the musical stimuli. In other words, there is more to listening to music than simply listening. Of course, the auditory process of how we physiologically hear sound is a significant part of the music experience, but many other parts of the brain and body become engaged as well, such as the emotional system, the language system, and the sensorimotor system. The first thing that usually comes to mind when thinking about how music affects the motor system is dancing, especially how music has the power to make us move almost instinctually. In fact, it was shown that certain types of music allow for synchronous rhythmic entrainment of the neurons in the auditory cortex, which manifests itself as head-bobbing or toe-tapping (Janata et al., 2007). The Janata et al. study investigated the concept of the "groove" in music, and specifically what makes certain music more "groovy" than other music. The study analyzed 148 songs and found that groovy songs fit within a tempo "sweet spot" and had high rhythmic syncopation. These conclusions gave me an idea: can I apply this "groovy" classification to the Spotify database of 160,000 songs?

As I mentioned above, music has myriad applications from the personal, to the academic, to the commercial. Likewise, this project that classifies music as "groovy" based on certain criteria established through empirical research is not only useful for people like me who want to create playlists of groovy music, but it also has plenty of potential in the commercial realm. Dance clubs and party venues benefit greatly from keeping people happy and energetic, and having access to a model that takes in music and outputs the songs that qualify as groovy would be a powerful tool. Hence, I used data science and classification models to create a groovy music classifier application.

___

## Story

To begin my analysis, I downloaded a Kaggle dataset of songs amassed from the Spotify database. The dataset contained 160,000 songs, their titles, artists, and other features such as tempo and danceability. The academic study found that the "groovy" songs had a tempo between 90 and 120 beats per minute and had strongly syncopated rhythms. I created my target of groovy by thresholding tempo and danceability to match the academic study’s standard, and classified the target using additional Spotify features such as duration, acousticness, release year, and so on. After creating the target, I found that about a fifth of the database’s songs met the groovy standard. Since I wanted to create a model that would take in a new song, analyze its features, and classify it as groovy or not, I needed to balance out the groovy-to-not-groovy ratio in my original database. This is because rather than learning how recognize a groovy song, the model would likely recognize that if it simply returns "not groovy" for every single song it gets, it will be correct 80% of the time. While an 80% success rate is not bad, having an application that simply returns "not groovy" 100% of the time is useless. Therefore, I oversampled to deal with this imbalance, and once the classes of "groovy" and "not groovy" were balanced, I ran 7 different machine learning classification models.

![Class Image](https://media-exp1.licdn.com/dms/image/C4E12AQFG6Mf-T8rtPg/article-inline_image-shrink_1000_1488/0/1614709408498?e=1627516800&v=beta&t=zl-AVeW46zrFvlyNSd6O3-qbqmWj07bvWyVo7E6lIzc)

To my disappointment, none of the models were great at consistently classifying songs correctly. The ROC Scores for the models reflect their poor accuracy overall, but the models were all still better than just guessing. The random forest model had the best test accuracy, but was still not perfect; it encouragingly returned over a thousand true positives, but unfortunately returned almost the same amount of false positives, resulting in a precision of .55. While this may look bad, it actually is not the end of the world. I realized that I could use my domain knowledge in music to think about this problem logically. I realized that the main goal should not be to identify every single “groovy” song in the entire database. I had to narrow my scope to get rid of the false positives and focus on precision, because clients would not want non-groovy music coming out of the model. I knew that I could get away with narrowing my scope and sacrificing recall because even if the model only returned 200 songs out of the thousands possible, that would still be more than enough for my client's needs, because the typical venue does not need 16 unique hours of music. To accomplish this, I increased the classification threshold to .7. This gave me a model that does miss a fair amount of groovy music, but is very rarely wrong about the music it deems to be worthy of the groovy classification.

![ROC Image](https://media-exp1.licdn.com/dms/image/C4E12AQFJVCmsf6IYaQ/article-inline_image-shrink_1000_1488/0/1614709493113?e=1627516800&v=beta&t=9ndietkHCCUsjFNA5YQcSVgdCjMMA75TqIgqs9-XpH4)

___


## Application

After I finalized my model, I got to work designing a web application that takes in user-submitted music and returns the songs that are indeed groovy. To accomplish this, I manually implemented the Spotify API in order to allow the user to view not just the titles of the groovy songs that come out of the model, but also Spotify musical features such as loudness or popularity. To test my application, I tried inputting an artist. Stevie Wonder’s "Superstition" was the most groovy song from the Janata et al. study, so I entered Stevie Wonder. The application returned a list of his groovy songs, and sure enough, "Superstition" was at the top of the list, followed by "Sir Duke". The app also provides each song's album name and the release date. I hope that people will find this app useful and have some fun exploring it. I also truly believe that DJs and business owners can employ this app to add some scientific backing and confidence to their music selection process.

![App Image](https://media-exp1.licdn.com/dms/image/C4E12AQEVBQ_lQc_Ipw/article-inline_image-shrink_1000_1488/0/1614727617735?e=1627516800&v=beta&t=FHT6zF3w7JjvPrrEMzFCFWynIM1fNyV8HplZq6EnOq4)

___

## Future Consideration

Looking forward, obviously precision in a model such as this one is important, but it can be limited if one does not have a large dataset to begin with. The best way to improve the model would be to include more features. As a musician, I would think that features such as bass repetitiveness and specific timbre would have an impact on how the brain entrains to rhythm. I would also love to expand my application to allow one to plug in a playlist, and have the application spit out a new playlist of just the groovy songs; perhaps I will call it "Groovyfy". Lastly, I am confident that music absolutely has a place in data science, just as data science has a place in music. The applications for both of these fields are endless; I hope to continue making strides to combine these two fields, and I encourage others to embrace the science in their music and the art in their data science.

___

## Author Information

- LinkedIn - [ZacharyMBrandt](https://www.linkedin.com/in/zacharymbrandt/)
- Website - [The Aspiring Music Psychologist](https://www.theaspiringmusicpsychologist.com)
- Podcast - [The Aspiring Music Psychologist](https://anchor.fm/zachary-brandt5)
