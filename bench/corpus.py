"""Sentences for load tests, shaped like what the Bun API sends: one sentence
per request, 40-250 characters, narration/explainer register."""

EN: tuple[str, ...] = (
    "The museum opens at 9:30 a.m. on weekdays.",
    "Why do some bridges sway in the wind while others stand perfectly still?",
    "In 1969, roughly 600 million people watched Neil Armstrong step onto the surface of the Moon.",
    "The recipe calls for 250 grams of flour, two eggs, and a pinch of salt.",
    "Dr. Patel had spent nearly a decade studying how coral reefs recover after a bleaching event, "
    "and her results surprised almost everyone in the field.",
    "Compound interest works quietly: a deposit of $1,000 growing at 5% a year roughly doubles in about fourteen years.",
    "She closed the notebook, looked out at the rain, and decided the letter could wait until morning.",
    "The Amazon River discharges more water than the next seven largest rivers combined.",
    "What happens to your body when you stop drinking coffee for a week?",
    "The U.S. Census Bureau estimates that the population passed 335 million in 2023.",
    "Most people underestimate how much light pollution affects sleep, mood, and even the migration "
    "patterns of birds that travel at night.",
    "Turn left at the second traffic light, then continue for about 400 meters.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen, and it is the "
    "foundation of nearly every food chain on Earth.",
    "He had never been on a train before, so every station felt like the edge of a new country.",
    "The company reported revenue of $4.2 billion for Q3, up 12% from the same quarter last year.",
    "Is it possible to learn a new language as an adult and still sound like a native speaker?",
    "The old lighthouse keeper climbed the 117 steps every evening, long after the lamp had been automated, "
    "simply because he liked the view from the top.",
    "Honey never spoils; archaeologists have found jars in Egyptian tombs that are over 3,000 years old.",
    "Before you begin, make sure the device is unplugged and the battery is fully removed.",
    "The James Webb Space Telescope orbits the Sun about 1.5 million kilometers from Earth, at a point "
    "where the gravity of both bodies keeps it in a stable position.",
    "Mr. and Mrs. Alvarez moved to Chicago in the spring of 1987.",
    "A single teaspoon of healthy soil can contain more living organisms than there are people on the planet.",
    "Our flight was delayed by three hours, which meant we arrived in Lisbon just after midnight.",
    "The first rule of good writing is simple: say what you mean, and then stop.",
    "When the Roman Empire was at its height, around 117 A.D., it stretched from northern England to the "
    "deserts of Syria and governed perhaps 60 million people.",
    "How long would it take to walk around the entire equator without stopping?",
    "At 3 o'clock sharp, the orchestra began tuning, and the hall slowly fell silent.",
    "Researchers at MIT found that short breaks every 90 minutes improved focus by as much as 20 percent.",
    "The village had one bakery, one church, and a single road that disappeared into the hills.",
    "Electric cars are often described as having zero emissions, but the full picture depends on how the "
    "electricity is generated and how the batteries are made.",
    "Please keep your seatbelt fastened while the captain has the sign illuminated.",
    "The Great Wall of China is not a single wall at all, but a network of fortifications built and rebuilt "
    "over roughly two thousand years.",
    "Sleep is when the brain clears out metabolic waste, consolidates memories, and resets the chemical "
    "balance that keeps our mood stable.",
    "Why does the sky turn red and orange at sunset?",
    "Blue whales can weigh up to 150 tonnes, and their hearts alone are about the size of a small car.",
    "Grandma always said that the secret to good soup was patience, and she was right.",
    "In chess, there are more possible games than there are atoms in the observable universe, which is why "
    "no computer has ever fully solved it.",
    "The bus leaves at 7:45, so we should be out the door by 7:30 at the latest.",
    "Vitamin D is produced in the skin when it is exposed to sunlight, yet many people living in northern "
    "countries are deficient during the winter months.",
    "It was the quietest summer anyone could remember, and nobody quite knew why.",
    "Mount Everest grows by about four millimeters every year as the Indian plate keeps pushing north.",
    "Did the ancient Greeks really believe that the Earth was flat?",
    "Inflation measures how quickly prices rise over time, and central banks usually aim to keep it close "
    "to 2% a year.",
    "The detective noticed that the clock on the mantel had stopped at exactly 11:52.",
    "Octopuses have three hearts, blue blood, and nine brains, if you count the clusters of neurons in each arm.",
    "If you water the plant once a week and keep it out of direct sunlight, it should flower by early June.",
    "The printing press, developed by Johannes Gutenberg around 1440, made books cheaper and faster to "
    "produce, and it helped spread ideas across Europe within a few decades.",
    "Our next stop is the National Gallery, which houses more than 2,300 paintings.",
    "She had read the same paragraph four times before admitting that she was simply too tired to understand it.",
    "Lightning strikes the Earth about 100 times every second.",
    "A good night's sleep, a short walk, and a glass of water will fix more problems than most people expect.",
    "Can a city really run entirely on renewable energy, or is that still decades away?",
    "The hotel's Wi-Fi password is printed on the back of your key card.",
    "During the Industrial Revolution, the population of Manchester grew from about 25,000 in 1772 to more "
    "than 300,000 by 1850, transforming a market town into a crowded industrial city.",
    "The river froze solid that winter, and children skated where boats had sailed only weeks before.",
    "Bananas are technically berries, while strawberries, despite their name, are not.",
    "Before the invention of the mechanical clock, most towns relied on church bells, sundials, and the "
    "position of the Sun to keep track of the hour.",
    "Remember to save your work every few minutes; the system restarts automatically at 2 a.m.",
    "Water boils at 100 degrees Celsius at sea level, but on top of Mount Everest it boils at only about 70.",
    "The human eye can distinguish roughly ten million different colors, yet the words we use to describe "
    "them vary enormously between languages, and some cultures traditionally did not separate blue and green.",
    "When the first transatlantic telegraph cable was completed in 1858, Queen Victoria's 98-word greeting "
    "to President Buchanan took more than sixteen hours to transmit, and the cable failed three weeks later.",
    "Long before satellites and weather radar, sailors read the sky for warnings: a ring around the Moon, "
    "a sudden drop in the barometer, or a red sunrise could mean a storm was less than a day away, and every "
    "experienced captain knew the signs by heart.",
    "And so, after forty years at sea, the captain finally returned home to the small harbor where his "
    "journey had begun.",
)

ES: tuple[str, ...] = (
    "El tren con destino a Sevilla saldrá del andén cuatro a las 8:15 de la mañana.",
    "La biblioteca municipal cuenta con más de 120.000 libros, y la mayoría se pueden pedir prestados gratis.",
    "¿Sabías que el corazón de una ballena azul puede pesar tanto como un coche pequeño?",
    "Durante el verano, el pueblo se llena de visitantes que vienen a disfrutar de sus playas tranquilas "
    "y de su comida casera.",
    "Mi abuela siempre decía que la paciencia es el ingrediente secreto de cualquier buena receta.",
)

FR: tuple[str, ...] = (
    "Le musée ouvre ses portes à neuf heures et demie, du mardi au dimanche.",
    "La tour Eiffel mesure environ 330 mètres et attire près de sept millions de visiteurs chaque année.",
    "Pourquoi le ciel devient-il rouge et orange au coucher du soleil ?",
    "Elle referma le livre, regarda la pluie tomber sur les toits, et décida que la lettre pouvait "
    "attendre le lendemain.",
    "Pour réussir cette recette, il faut 250 grammes de farine, trois œufs et une pincée de sel.",
)

DE: tuple[str, ...] = (
    "Der Zug nach München fährt heute ausnahmsweise von Gleis sieben ab.",
    "Die Donau fließt durch zehn Länder und mündet nach fast 2.900 Kilometern ins Schwarze Meer.",
    "Wie lange würde es dauern, zu Fuß einmal um den ganzen Äquator zu laufen?",
    "Nach dem langen Winter freuten sich alle auf die ersten warmen Tage im Garten.",
    "Bitte speichern Sie Ihre Arbeit regelmäßig, da das System um zwei Uhr nachts automatisch neu startet.",
)

# Chinese packs ~4x more speech per character, so these are shorter in chars.
ZH: tuple[str, ...] = (
    "今天下午三点，博物馆将举办一场关于古代丝绸之路的专题讲座，欢迎大家准时参加。",
    "长江全长约六千三百公里，是亚洲最长的河流，也是世界第三长河。",
    "你有没有想过，为什么日落时天空会变成红色和橙色？",
    "她合上笔记本，望着窗外的细雨，决定把那封信留到明天早上再写。",
    "研究表明，每天散步三十分钟，可以明显改善睡眠质量和情绪。",
)

CORPUS: dict[str, tuple[str, ...]] = {"en": EN, "es": ES, "fr": FR, "de": DE, "zh": ZH}
