# natural-language-processing

Het geautomatiseerd verwerken, analyseren en genereren van menselijke taal.
Door middel van tokenisatie, bag-of-words classificatie, n-gram tekstgeneratie en
embeddings. 

- Tokenisatie:

Tokenisatie is het proces waarin een tekst wordt omgezet in een opeenvolgende reeks tokens die kunnen worden 
geidentificeerd met een numeriek label (d.w.z. een integer). Tokenisatie helpt modellen bij het verwerken van grote
hoveeelheden data door tekst op te splitsen in kleinere eenheden. Dit maakt het mogelijk om patronen en verbanden te 
leren, wat de prestaties en nauwkeurigheid verbetert.

- Bag-of-words classificatie:

Bag-of-words classificatie is een techniek om tekst om te zetten in een numerieke representatie (een zak met woorden)
door de frequentie van elk woord of kenmerk te tellen, zonder rekening te houden met de volgorde, en deze vectoren
vervolgens te gebruiken om documenten te classificeren(bijvoorbeeld spam filteren) met machine learning-algoritmes.

- N-gram tekstgeneratie:

N-gram tekstgeneratie is een klassieke methode om tekst te voorspellen of te genereren op basis van frequenties 
van opeenvolgende woorden of karakters. Een N-gram is een opeenvolging van N woorden (of karakters) in een tekst.
Zo heb je bijvoorbeeld een unigram waatbij een zin opgesplist wordt in enkele woorden. En een trigram waarbij er
3 woorden achter elkaar staan. Hoe hoger N, hoe beter de context wordt behouden, maar hoe meer data je nodig hebt.

- Embeddings:

Embeddings zijn een manier om woorden, zinnen of andere entiteiten om te zetten naar getallen in een vectorruimte.
Ze zijn vectors van getallen, meestal van vaste lengte (bijvoorbeeld 50, 100 of 768 dimensies).
Het doel: de betekenis van woorden wiskundig representeren, zodat woorden met vergelijkbare betekenis dicht bij elkaar 
liggen in die ruimte. Stel je voor dat elk woord een punt is op een kaart. Woorden met vergelijkbare betekenis staan 
dicht bij elkaar.
