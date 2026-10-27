# Vervolgvragen voor de vakman / Pytania dla fachowca (2026-10-26)
Per vraag in [haken] waar het in okc voor staat (alleen voor ons; niet voorlezen).

## A. Helpers en de vakman / Pomocnicy i fachowiec  [MoE-kernel: warp specialisation, residency, skew, split-K]
1. NL: Als de loodgieter alvast de leidingen legt voor de elektricien: werkt de elektricien intussen in een andere kamer, of wacht hij tot het voorwerk klaar is? Hoeveel helpers zet je maximaal bij één vakman voordat ze elkaar in de weg lopen?
   PL: Kiedy hydraulik już przygotowuje rury dla elektryka: czy elektryk w tym czasie pracuje w innym pokoju, czy czeka, aż przygotowanie będzie gotowe? Ilu pomocników maksymalnie dajesz do jednego fachowca, zanim zaczną sobie przeszkadzać?
   [producer/consumer verhouding; wanneer extra warps elkaar hinderen]
2. NL: Breng je bij een grote klus al het materiaal voor de hele dag in één keer naar binnen, of laat je het in porties aanleveren? Wat doe je als de kamer te klein is voor de hele stapel?
   PL: Przy dużym zleceniu: wnosisz od razu cały materiał na cały dzień, czy zamawiasz dostawy w porcjach? Co robisz, gdy pokój jest za mały na cały stos?
   [shared-memory voorraad vs. aantal ploegen per SM — de reden dat de "ring" verloor]
3. NL: Als je maar één erkende vakman hebt en veel kamers: zoek je een snellere vakman, of haal je elke simpele stap uit zijn werk en geef je die aan anderen? Welke stappen kan een helper veilig vooraf doen?
   PL: Jeśli masz tylko jednego uprawnionego fachowca i wiele pokoi: szukasz szybszego fachowca, czy wyciągasz z jego pracy każdy prosty krok i dajesz go innym? Które kroki pomocnik może bezpiecznie zrobić wcześniej?
   [welk deel van dequant/exp naar "algemene" rekeneenheden kan]
4. NL: Stel: één kamer heeft tien keer zoveel werk als de rest (één badkamer vol, de andere kamers bijna leeg). Zet je dan meer mensen in die ene kamer, of doet één ploeg die kamer terwijl de rest de kleine kamers afmaakt? En wie doet de afwerking als meerdere ploegen aan dezelfde muur werkten?
   PL: Załóżmy, że jeden pokój ma dziesięć razy więcej pracy niż reszta (jedna łazienka pełna roboty, inne pokoje prawie puste). Dajesz więcej ludzi do tego jednego pokoju, czy jedna ekipa robi ten pokój, a reszta kończy małe pokoje? I kto robi wykończenie, jeśli kilka ekip pracowało przy tej samej ścianie?
   [één "hete" expert met 77 tokens; split-K en wie de deelsommen samenvoegt]

## B. Verspreide materialen / Rozproszone materiały  [attention: 80.000 losse KV-rijen, exp-knelpunt]
5. NL: Stel dat je voor één klus 80.000 losse onderdelen moet ophalen, elk bij een ander adres in de stad, maar je kent de adressen vooraf. Hoe organiseer je dat: één bus die alles afrijdt, veel fietsers tegelijk, of bundelen per wijk? Wanneer helpen extra fietsers niet meer?
   PL: Wyobraź sobie, że do jednego zlecenia musisz odebrać 80 000 drobnych części, każdą pod innym adresem w mieście, ale adresy znasz z góry. Jak organizujesz odbiór: jeden bus, który objeżdża wszystko, wielu rowerzystów naraz, czy grupujesz według dzielnic? Kiedy dodatkowi rowerzyści już nie pomagają?
   [paginated KV met page size 1: hoeveel gelijktijdige "ophalers" per SM]
6. NL: De vakman die elk onderdeel moet keuren is het knelpunt: hij moet elk stuk vastpakken en beoordelen. Zijn er keuringen die een helper met een simpele meetlat bijna net zo goed kan doen, zodat de vakman alleen de lastige gevallen pakt?
   PL: Fachowiec, który musi sprawdzić każdą część, jest wąskim gardłem: każdą musi wziąć do ręki i ocenić. Czy są sprawdzenia, które pomocnik z prostą miarką zrobi prawie tak samo dobrze, żeby fachowiec brał tylko trudne przypadki?
   [exp2 deels via polynoom op FMA-eenheden, de MUFU alleen voor de rest]

## C. De volgorde van vakmensen / Kolejność fachowców  [decoder layer: ~20 stappen achter elkaar]
7. NL: Bij een verbouwing wacht elke vakman op de vorige (eerst sloop, dan leidingen, dan stucwerk, dan de schilder). Hoe verkort je zo'n keten in de praktijk? Laat je de volgende vakman al beginnen in een deel van het huis terwijl de vorige nog bezig is? Waar gaat dat mis?
   PL: Przy remoncie każdy fachowiec czeka na poprzedniego (najpierw rozbiórka, potem instalacje, potem tynk, potem malarz). Jak w praktyce skracasz taki łańcuch? Czy pozwalasz następnemu zacząć w części domu, kiedy poprzedni jeszcze pracuje? Gdzie to się psuje?
   [kernels laten overlappen per deel van de data; afhankelijkheden]
8. NL: Hoeveel tijd verlies je aan het wisselen tussen vakmensen (aankomen, gereedschap uitpakken, werk overdragen), en hoe beperk je dat?
   PL: Ile czasu tracisz na zmianę fachowców (dojazd, rozpakowanie narzędzi, przekazanie pracy) i jak to ograniczasz?
   [launch-gaps en "opstart" van elke kernel]

## D. Heel kleine klusjes / Bardzo małe roboty  [bf16: GEMV's van een paar microseconden]
9. NL: Bij een heel klein klusje (één stopcontact) kost rijden en uitpakken meer tijd dan het werk zelf. Combineer je dat met andere klusjes, of laat je één persoon een hele reeks achter elkaar doen? Wanneer loont dat niet meer?
   PL: Przy bardzo małej robocie (jedno gniazdko) dojazd i rozpakowanie zajmują więcej czasu niż sama praca. Łączysz to z innymi robotami, czy jedna osoba robi całą serię po kolei? Kiedy to się już nie opłaca?
   [kleine kernels samenvoegen / één kernel die meerdere stappen doet]

## E. Leren en stoppen / Nauka i kiedy przestać  [methode]
10. NL: Hoe kom je erachter welke standaardmaten en gereedschappen er bestaan die je nog niet kent (zoals die 140-dorpels)? Wie vraag je als eerste?
    PL: Jak dowiadujesz się, jakie standardowe wymiary i narzędzia istnieją, których jeszcze nie znasz (jak te progi 140)? Kogo pytasz najpierw?
    [Hopper-functies die we nog niet gebruiken; waar leer je ze]
11. NL: Hoeveel balken bekijk je voordat je stopt met kiezen: een vast aantal, of tot je genoeg hebt gezien die recht zijn aan de kant die telt?
    PL: Ile belek oglądasz, zanim przestajesz wybierać: stałą liczbę, czy aż zobaczysz wystarczająco dużo prostych z tej strony, która się liczy?
    [stopregel voor varianten zoeken]
12. NL: Wanneer weet je dat een aanpak die je al twee keer hebt geprobeerd echt niet gaat werken, en wanneer probeer je het een derde keer op een andere manier?
    PL: Kiedy wiesz, że sposób, który próbowałeś już dwa razy, naprawdę nie zadziała, a kiedy próbujesz trzeci raz inaczej?
    [onze twee-strikes-regel: wanneer toch doorzetten]
