# Word2Vec-CPP
An implementation of Word2Vec in CPP.

This codebase was created by migrating the original Word2Vec code from Google (https://github.com/tmikolov/word2vec) from C to C++.
It's using the machine learning library I built while working at Fetch.ai (https://github.com/fetchai/ledger) with some modification to improve performances.

# Usage

I recomend using the text8 training corpus from Matt Mahoney. It's about 100MB of data, and it's been proven to produce decent embeddings.

```wget http://mattmahoney.net/dc/text8.zip```

Build with Cmake ```mkdir build && cd build && cmake .. && make```

Run using ```./Word2Vec ../text8```
This should run for about 18 minutes (9 per epoch) on a Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz

This will save the trained embeddings as ```vector.bin```

You can then check the results using ```./Distance vector.bin```
The distance.c code is a copy & paste from the original Google codebase as saving format is the same.

# Sample output - 10 epochs (1.5 hours)

```
Enter word or sentence (EXIT to break): france

Word: france  Position in vocabulary: 24660

                                              Word       Cosine distance
------------------------------------------------------------------------
                                             spain		0.866192
                                             italy		0.826700
                                           germany		0.766490
                                           belgium		0.760248
                                          portugal		0.738122
                                            russia		0.709743
                                           england		0.704960
                                          provence		0.703192
                                           britain		0.698285
                                            poland		0.687686
                                           austria		0.682699
                                            sicily		0.675318
                                             paris		0.674724
                                            greece		0.673848
                                        luxembourg		0.669873
                                          flanders		0.669805
                                           tripoli		0.669777
                                          brittany		0.663397
                                            sweden		0.663067
                                           denmark		0.657452
                                      transylvania		0.655827
                                            naples		0.655381
                                           finland		0.653961
                                           hungary		0.651292
                                           morocco		0.650929
                                            calais		0.647290
                                         argentina		0.646196
                                              metz		0.645684
                                       switzerland		0.645662
                                           seville		0.642026
                                             tunis		0.640765
                                          scotland		0.634927
                                            europe		0.632660
                                          salzburg		0.630270
                                         catalonia		0.629717
                                          normandy		0.625913
                                           corsica		0.625167
                                          toulouse		0.622875
                                          sardinia		0.622850
                                            brazil		0.622817
Enter word or sentence (EXIT to break): audi

Word: audi  Position in vocabulary: 4684

                                              Word       Cosine distance
------------------------------------------------------------------------
                                               bmw		0.667833
                                        volkswagen		0.634313
                                              benz		0.627654
                                             mazda		0.623202
                                          mercedes		0.611886
                                          chrysler		0.608197
                                             miata		0.604863
                                            ducati		0.602944
                                            toyota		0.591280
                                               dkw		0.591215
                                            marque		0.581741
                                               bsa		0.571802
                                           bugatti		0.564583
                                           midsize		0.551619
                                           daimler		0.546291
                                              jeep		0.546114
                                          vauxhall		0.545731
                                             busch		0.539897
                                            ibanez		0.538711
                                           motoren		0.538212
                                            monaro		0.537962
                                          roadster		0.537542
                                           renault		0.532050
                                        koenigsegg		0.531523
                                             lexus		0.531341
                                             coupe		0.530718
                                             honda		0.530714
                                        automobile		0.528760
                                           engined		0.526645
                                          anheuser		0.526567
                                       lamborghini		0.526240
                                          cadillac		0.526061
                                           porsche		0.525039
                                           peugeot		0.523487
                                             atari		0.520767
                                              saab		0.520219
                                   daimlerchrysler		0.516619
                                           mclaren		0.511398
                                           hyundai		0.509458
                                          scuderia		0.506369
Enter word or sentence (EXIT to break): beer

Word: beer  Position in vocabulary: 6284

                                              Word       Cosine distance
------------------------------------------------------------------------
                                              malt		0.745644
                                            cheese		0.738951
                                             lager		0.728271
                                             cider		0.715501
                                         fermented		0.714066
                                           draught		0.712888
                                               ale		0.695219
                                              wine		0.685727
                                           brewing		0.674713
                                           tasting		0.666819
                                             beers		0.666147
                                             vodka		0.656985
                                         chocolate		0.655863
                                            brewed		0.648331
                                             stout		0.646192
                                           liqueur		0.642679
                                              bock		0.641109
                                             drink		0.640543
                                          beverage		0.639973
                                            liquor		0.636998
                                              ales		0.632753
                                          absinthe		0.619993
                                             cream		0.617283
                                              milk		0.617180
                                            whisky		0.606963
                                           pilsner		0.605299
                                              cask		0.604258
                                            pastry		0.599440
                                           vinegar		0.595087
                                        carbonated		0.593672
                                          desserts		0.592873
                                         beverages		0.592373
                                       carbonation		0.589116
                                            baking		0.589025
                                            drinks		0.586615
                                               gin		0.583482
                                           bottles		0.577443
                                         rauchbier		0.576952
                                             syrup		0.574558
                                               keg		0.573618
```
