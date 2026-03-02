"""
Restaurant Dataset Generator
Generates 300 diverse synthetic restaurants for the recommendation engine.
"""

import json
import random

random.seed(42)

CUISINES = [
    "Italian", "Chinese", "Japanese", "Mexican", "Indian",
    "Thai", "French", "American", "Mediterranean", "Korean",
    "Vietnamese", "Greek", "Spanish", "Middle Eastern", "Ethiopian",
    "Brazilian", "Peruvian", "Lebanese", "Turkish", "Caribbean"
]

AMBIANCES = ["Casual", "Fine Dining", "Family-Friendly", "Romantic", "Trendy", "Cozy", "Lively", "Quiet"]
LOCATIONS = [
    "Downtown", "Midtown", "East Side", "West Side", "Waterfront",
    "Arts District", "University Area", "Chinatown", "Little Italy",
    "Financial District", "Suburbs", "North End", "South End"
]
PRICE_RANGES = ["$", "$$", "$$$", "$$$$"]
DIETARY = ["Vegetarian-friendly", "Vegan options", "Gluten-free options", "Halal", "Kosher", "Dairy-free options"]
FEATURES = ["Takeout", "Delivery", "Dine-in", "Outdoor seating", "Reservations accepted",
            "Bar", "Private dining", "Live music", "Happy hour", "Parking available", "Pet-friendly"]

CUISINE_DATA = {
    "Italian": {
        "dishes": ["Margherita Pizza", "Cacio e Pepe", "Osso Buco", "Tiramisu", "Risotto al Tartufo",
                   "Lasagna Bolognese", "Seafood Linguine", "Gnocchi al Pesto", "Saltimbocca", "Cannoli",
                   "Bruschetta", "Penne Arrabbiata", "Ribollita", "Panzanella", "Pappardelle al Cinghiale"],
        "descriptions": [
            "A charming trattoria serving authentic Neapolitan dishes with ingredients imported weekly from Italy. The wood-fired oven produces perfectly charred pizzas and the homemade pasta is prepared fresh daily.",
            "An upscale Italian restaurant known for its regional Italian cuisine and extensive wine cellar. Chef's specialties include hand-rolled pasta and slow-braised meats that reflect generations of culinary tradition.",
            "A cozy neighborhood Italian spot where the aroma of garlic and olive oil fills the air. The menu rotates seasonally to highlight the freshest local produce paired with classic Italian techniques.",
            "An elegant ristorante offering fine Italian dining with tableside preparations and an award-winning sommelier. The intimate atmosphere and impeccable service make every meal a memorable occasion.",
            "A family-run Italian kitchen that has been serving the community for over 20 years. Recipes passed down through three generations ensure an authentic taste of the Italian countryside.",
        ]
    },
    "Chinese": {
        "dishes": ["Dim Sum", "Peking Duck", "Kung Pao Chicken", "Mapo Tofu", "Xiaolongbao",
                   "Char Siu", "Wonton Soup", "Dan Dan Noodles", "Scallion Pancakes", "Beef with Broccoli",
                   "Hot and Sour Soup", "General Tso's Chicken", "Pork Belly Bao", "Shrimp Dumplings", "Twice-Cooked Pork"],
        "descriptions": [
            "A bustling Cantonese restaurant specializing in traditional dim sum served from roaming carts on weekend mornings. The menu also features classic Cantonese barbecue and hearty seafood dishes.",
            "An authentic Sichuan restaurant that brings the bold, numbing heat of southwestern China to every dish. The peppercorn-laden specialties and hand-pulled noodles are prepared by chefs trained in Chengdu.",
            "A modern Chinese bistro blending traditional techniques with contemporary plating. The open kitchen lets diners watch skilled chefs craft delicate dumplings and masterfully stir-fry each dish to order.",
            "A celebrated restaurant known for its ceremonial Peking Duck carved tableside and the extensive menu of regional Chinese dishes from across eight culinary regions.",
            "A family-style Chinese restaurant where sharing is encouraged and portions are generous. The menu covers classic Cantonese, Shanghainese, and Hakka dishes all made from scratch.",
        ]
    },
    "Japanese": {
        "dishes": ["Omakase Sushi", "Tonkotsu Ramen", "Wagyu Beef Teppanyaki", "Karaage Chicken",
                   "Tempura Udon", "Chirashi Bowl", "Yakitori", "Matcha Desserts", "Gyoza",
                   "Takoyaki", "Miso Black Cod", "Katsu Curry", "Edamame", "Salmon Sashimi", "Tamagoyaki"],
        "descriptions": [
            "A serene Japanese restaurant offering an intimate omakase experience led by a master sushi chef. Each piece is hand-crafted with premium fish flown in directly from Tsukiji market.",
            "A lively izakaya-style restaurant where small plates and sake flow freely. The menu features grilled skewers, crispy tempura, and inventive Japanese pub fare in a convivial setting.",
            "A sleek ramen shop specializing in rich, 18-hour slow-cooked broths. From silky tonkotsu to tangy shio, each bowl is a labor of love topped with house-made noodles and premium toppings.",
            "A modern Japanese restaurant combining traditional techniques with seasonal local ingredients. The chef's creative kaiseki-inspired tasting menu changes monthly to reflect nature's rhythm.",
            "A casual Japanese eatery beloved for its crispy tonkatsu and comforting curry rice. Quick service and generous portions make it a neighborhood favorite for lunch and dinner.",
        ]
    },
    "Mexican": {
        "dishes": ["Carnitas Tacos", "Chiles en Nogada", "Mole Negro", "Tamales", "Enchiladas Verdes",
                   "Ceviche", "Guacamole", "Pozole Rojo", "Cochinita Pibil", "Churros con Chocolate",
                   "Elote", "Birria Tacos", "Tlayuda", "Chiles Rellenos", "Agua Fresca"],
        "descriptions": [
            "A festive Mexican cantina serving bold, regional recipes from Oaxaca and Veracruz. The house-made moles and slow-cooked meats honor centuries of culinary tradition in a vibrant, colorful setting.",
            "A modern taqueria with an extensive selection of artisanal tacos, each served on fresh hand-pressed corn tortillas. The salsa bar features 12 varieties made daily from fire-roasted ingredients.",
            "An upscale Mexican restaurant that elevates traditional dishes with refined techniques and premium ingredients. The tasting menu showcases the biodiversity of Mexican cuisine beyond the usual suspects.",
            "A casual neighborhood Mexican spot known for its generous portions, fresh ingredients, and unbeatable margaritas. The friendly staff and lively atmosphere make it perfect for group gatherings.",
            "A family-owned Mexican kitchen where grandma's recipes shine through in every dish. The slow-cooked stews, hand-rolled tamales, and house-made tortillas reflect the warmth of home cooking.",
        ]
    },
    "Indian": {
        "dishes": ["Butter Chicken", "Dal Makhani", "Lamb Biryani", "Palak Paneer", "Garlic Naan",
                   "Tandoori Chicken", "Samosa Chaat", "Masala Dosa", "Gulab Jamun", "Chicken Tikka Masala",
                   "Chole Bhature", "Rogan Josh", "Kheer", "Pav Bhaji", "Malai Kofta"],
        "descriptions": [
            "An aromatic Indian restaurant transporting diners to the spice markets of Mumbai and Delhi. The tandoor oven produces beautifully charred breads and meats while the kitchen simmers rich curries made with freshly ground spices.",
            "A South Indian vegetarian restaurant specializing in crispy dosas, fluffy idlis, and fragrant sambars. The coconut chutneys and lentil-based dishes represent the lighter, vibrant cuisine of Kerala and Tamil Nadu.",
            "A fine-dining Indian restaurant that reinterprets classic dishes through a modern lens. The chef's tasting menu features unexpected flavor combinations while preserving the soul of traditional Indian cooking.",
            "A beloved North Indian restaurant known for its creamy, slow-cooked curries and pillowy breads fresh from the tandoor. The lamb and paneer dishes are particularly celebrated by regulars.",
            "A casual Indian eatery where the buffet lunch is legendary in the neighborhood. The rotating selection of curries, biryanis, and vegetarian dishes ensures something new to discover every visit.",
        ]
    },
    "Thai": {
        "dishes": ["Pad Thai", "Green Curry", "Tom Yum Soup", "Som Tum", "Massaman Curry",
                   "Pad See Ew", "Larb", "Mango Sticky Rice", "Satay", "Pad Kra Pao",
                   "Khao Soi", "Thai Iced Tea", "Spring Rolls", "Pla Kapong", "Crying Tiger Beef"],
        "descriptions": [
            "A vibrant Thai restaurant balancing the four pillars of Thai flavor: sour, sweet, salty, and spicy. Fresh herbs, house-made curry pastes, and imported Thai ingredients define every dish on the menu.",
            "A casual Thai street food spot bringing Bangkok's bustling sidewalk flavors to the city. The wok-tossed noodles and fiery papaya salads are prepared with the authenticity of a Bangkok hawker stall.",
            "An upscale Thai restaurant presenting royal Thai cuisine with elegant plating and refined flavors. The complex, multi-ingredient dishes reflect the sophisticated palace cooking tradition of Thailand.",
            "A neighborhood Thai restaurant where regulars return weekly for the comforting curries and bold noodle dishes. The recipes come straight from the owner's family home in Chiang Mai.",
            "A modern Thai bistro blending traditional recipes with local seasonal ingredients. The innovative cocktail menu features Thai herbs and spices that complement the food beautifully.",
        ]
    },
    "French": {
        "dishes": ["Beef Bourguignon", "French Onion Soup", "Duck Confit", "Crème Brûlée", "Escargot",
                   "Bouillabaisse", "Steak Frites", "Soufflé au Chocolat", "Foie Gras", "Ratatouille",
                   "Coq au Vin", "Quiche Lorraine", "Crêpes Suzette", "Cassoulet", "Tarte Tatin"],
        "descriptions": [
            "A classic French bistro evoking the charm of a Parisian neighborhood restaurant. The menu features timeless brasserie dishes prepared with meticulous technique and the finest imported French ingredients.",
            "An intimate French restaurant where the chef brings his Lyonnaise training to bear on seasonal tasting menus. The wine list focuses exclusively on small-production French producers from underrated appellations.",
            "A casual French café perfect for leisurely lunches of croque monsieurs and afternoon pastries. The breakfast menu features authentic viennoiseries baked fresh each morning by our pastry chef.",
            "An elegant fine-dining restaurant offering contemporary French cuisine with a focus on local terroir. The modernist techniques applied to classic French flavor profiles create an exciting dining experience.",
            "A cozy French restaurant known for its hearty provincial cooking and generous pours of natural wine. The rustic, unfussy dishes like cassoulet and pot-au-feu bring the French countryside to your table.",
        ]
    },
    "American": {
        "dishes": ["Wagyu Smash Burger", "BBQ Brisket", "Lobster Roll", "Fried Chicken", "Clam Chowder",
                   "Mac and Cheese", "Buffalo Wings", "Club Sandwich", "Apple Pie", "Caesar Salad",
                   "Pulled Pork Sandwich", "Cornbread", "Fish and Chips", "Cheesesteak", "Pancake Stack"],
        "descriptions": [
            "A celebrated American grill showcasing the diversity of regional American cooking. From New England seafood to Texas BBQ, the menu celebrates the rich patchwork of American culinary traditions.",
            "A farm-to-table American restaurant sourcing ingredients from local farms within 100 miles. The seasonal menu changes weekly to reflect what's at peak freshness from our trusted farmer partners.",
            "A classic American diner serving generous portions of comfort food in a retro setting. The hand-pressed burgers, crispy fried chicken, and fluffy pancakes are made from scratch and consistently excellent.",
            "A modern American steakhouse featuring prime cuts dry-aged in-house for 28-45 days. The sommelier-curated wine program and house-made sides elevate the classic steakhouse experience.",
            "A casual American gastropub where craft beer and elevated bar food create the perfect combination. The rotating tap list of local brews pairs beautifully with the scratch-made wings and creative sandwiches.",
        ]
    },
    "Mediterranean": {
        "dishes": ["Mezze Platter", "Grilled Octopus", "Lamb Kofta", "Falafel", "Hummus",
                   "Moussaka", "Spanakopita", "Tabbouleh", "Baklava", "Fattoush",
                   "Shish Tawook", "Shakshuka", "Saganaki", "Loukoumades", "Dolmades"],
        "descriptions": [
            "A sun-drenched Mediterranean restaurant transporting diners to the shores of the Aegean. The mezze platters, fresh-grilled seafood, and vibrant vegetable dishes celebrate the wholesome Mediterranean diet.",
            "An authentic Lebanese restaurant where the menu spans the breadth of the Levantine table. Freshly baked pita, house-made dips, and slow-roasted meats create a feast meant for sharing.",
            "A Greek taverna with a lively atmosphere and generous portions of classic Greek dishes. The freshly made tzatziki, perfectly charred souvlaki, and flaky spanakopita are perennial favorites.",
            "A refined Mediterranean restaurant drawing inspiration from the cuisines of Spain, Italy, Greece, and North Africa. The open kitchen and wood-fire grill are the heart of the cooking philosophy.",
            "A casual Mediterranean café serving fresh, wholesome dishes throughout the day. The extensive vegetarian menu and bright, herb-forward flavors make it a healthy haven in the neighborhood.",
        ]
    },
    "Korean": {
        "dishes": ["Korean BBQ", "Bibimbap", "Sundubu Jjigae", "Japchae", "Tteokbokki",
                   "Kimchi Jeon", "Samgyeopsal", "Doenjang Jjigae", "Galbi", "Haemul Pajeon",
                   "Naengmyeon", "Bossam", "Jajangmyeon", "Army Stew", "Soju Cocktails"],
        "descriptions": [
            "A lively Korean BBQ restaurant where diners grill premium meats over charcoal tabletop grills. The extensive banchan selection and house-fermented kimchis make each meal a full sensory experience.",
            "A modern Korean restaurant presenting traditional dishes with contemporary flair. The chef's creative reinterpretations of classic jjigaes and rice dishes introduce Korean cuisine to new audiences.",
            "A cozy Korean homestyle restaurant serving the comforting stews, soups, and rice dishes of everyday Korean cooking. The rotating daily specials reflect the owner's childhood meals in Seoul.",
            "A trendy Korean fusion spot combining Korean flavors with global influences. The creative menu features Korean tacos, kimchi fried rice, and bulgogi burgers alongside traditional dishes.",
            "A family-run Korean restaurant known for its slow-simmered broths and house-made rice cakes. The weekend brunch menu features traditional Korean morning dishes not found elsewhere in the city.",
        ]
    },
    "Vietnamese": {
        "dishes": ["Pho Bo", "Banh Mi", "Fresh Spring Rolls", "Bun Bo Hue", "Crispy Pork Belly",
                   "Vietnamese Coffee", "Com Tam", "Bun Cha", "Goi Cuon", "Canh Chua",
                   "Banh Xeo", "Lemongrass Chicken", "Pho Ga", "Vietnamese Crepe", "Chao Tom"],
        "descriptions": [
            "A beloved Vietnamese restaurant famous for its soul-warming pho, simmered for 12 hours with charred ginger and aromatic spices. The extensive menu of noodle soups, rice dishes, and fresh rolls covers the breadth of Vietnamese cuisine.",
            "A hip Vietnamese café bringing the street food culture of Hanoi and Saigon to a trendy urban setting. The bánh mì sandwiches, fresh summer rolls, and Vietnamese iced coffees are particularly beloved.",
            "An authentic Vietnamese restaurant run by a family from Hội An, featuring the distinctive Central Vietnamese cuisine rarely found outside Vietnam. The complex flavors and fresh herb garnishes make each dish memorable.",
            "A fast-casual Vietnamese spot specializing in customizable bowls and noodle dishes made with fresh, locally sourced ingredients. The bold flavors and healthy preparations attract a loyal lunchtime crowd.",
            "A romantic Vietnamese restaurant with elegant plating and a focus on the refined dishes of the Vietnamese royal court. The tasting menu is a journey through the complexity and sophistication of Vietnamese culinary heritage.",
        ]
    },
    "Greek": {
        "dishes": ["Lamb Souvlaki", "Moussaka", "Spanakopita", "Grilled Octopus", "Tzatziki",
                   "Pastitsio", "Horiatiki Salad", "Loukoumades", "Kalamari", "Saganaki",
                   "Lamb Kleftiko", "Tiropita", "Avgolemono Soup", "Papoutsakia", "Galaktoboureko"],
        "descriptions": [
            "A lively Greek taverna where the food, wine, and spirit of the Greek islands come alive every evening. The mezze selections, grilled proteins, and warm hospitality create an authentic taste of Greece.",
            "An upscale Greek restaurant showcasing the sophisticated side of Hellenic cuisine. Beyond the familiar classics, the menu features regional specialties and wild-caught seafood prepared with Aegean inspiration.",
            "A casual Greek eatery owned by a family from Thessaloniki who brought their recipes and traditions overseas. The slow-roasted meats and handmade phyllo pastries reflect generations of Greek home cooking.",
            "A modern Greek bistro reinterpreting ancient flavors through a contemporary culinary lens. The seasonal menu incorporates local ingredients into quintessentially Greek preparations.",
            "A festive Greek restaurant where belly dancing, live music, and generous platters create a Mediterranean feast atmosphere. The family-style sharing menu encourages communal dining and celebration.",
        ]
    },
    "Spanish": {
        "dishes": ["Patatas Bravas", "Jamon Iberico", "Gambas al Ajillo", "Paella Valenciana",
                   "Croquetas", "Pulpo a la Gallega", "Tortilla Española", "Crema Catalana",
                   "Pimientos de Padron", "Albondigas", "Gazpacho", "Churros con Chocolate",
                   "Pan con Tomate", "Morcilla", "Sangria"],
        "descriptions": [
            "A vibrant Spanish tapas bar where small plates of authentic Spanish bites pair perfectly with Rioja and Cava. The social atmosphere, communal tables, and rotating menu of pintxos capture the spirit of a San Sebastián bar.",
            "An upscale Spanish restaurant focusing on the refined cuisine of Catalonia and the Basque Country. The tasting menu showcases avant-garde techniques alongside reverence for exceptional Spanish ingredients.",
            "A casual Spanish taverna specializing in generous paellas cooked to order in traditional pans. The socarrat-laden rice dishes and extensive sherry selection transport diners to the heart of Valencia.",
            "A lively Spanish restaurant and flamenco venue where passionate performances accompany the feast. The ibérico charcuterie, seafood tapas, and regional Spanish wines create an immersive Iberian experience.",
            "A neighborhood Spanish bar open late and beloved for its affordable tapas, excellent house wine, and convivial atmosphere. The kitchen's simplicity and quality ingredients reflect the best of everyday Spanish cooking.",
        ]
    },
    "Middle Eastern": {
        "dishes": ["Shawarma", "Falafel", "Hummus", "Fattoush", "Kibbeh",
                   "Mansaf", "Kebab", "Baklava", "Pita Bread", "Baba Ganoush",
                   "Musakhan", "Knafeh", "Mujaddara", "Ful Medames", "Za'atar Manakish"],
        "descriptions": [
            "A welcoming Middle Eastern restaurant where the flavors of the Levant fill the warm, hospitality-rich dining room. The slow-cooked meats, fresh-ground spices, and house-baked pita breads are highlights of the extensive menu.",
            "A modern Middle Eastern restaurant blending the diverse culinary traditions of Egypt, Lebanon, Syria, and Jordan. The mezze-style dining encourages exploration of the region's extraordinary range of flavors.",
            "A casual shawarma and falafel shop serving freshly made street food classics with exceptional quality. The slow-roasting spit and house-made sauces elevate these beloved dishes beyond the ordinary.",
            "An upscale Middle Eastern fine-dining establishment presenting the refined cuisine of the region with impeccable technique. The aromatic spice blends and intricate flavor layering reflect the ancient sophistication of Middle Eastern cookery.",
            "A family-run Middle Eastern restaurant where recipes from grandmother's kitchen in Amman are faithfully recreated. The generous portions, warm welcome, and authentic flavors make it a home away from home.",
        ]
    },
    "Ethiopian": {
        "dishes": ["Doro Wat", "Injera", "Kitfo", "Tibs", "Misir Wat",
                   "Gomen", "Shiro", "Azifa", "Ful", "Kategna",
                   "Firfir", "Ayib", "Tej", "Berbere Lamb", "Vegetarian Combo"],
        "descriptions": [
            "An authentic Ethiopian restaurant where communal dining on injera loaded with vibrant stews creates a unique and memorable experience. The aromatic berbere spice blends and complex wats reflect the depth of Ethiopian culinary heritage.",
            "A welcoming Ethiopian family restaurant serving traditional dishes in a warmly decorated dining room. The vegetarian platter with its rainbow of lentil, vegetable, and cheese dishes is celebrated by plant-based diners.",
            "A modern Ethiopian restaurant presenting traditional flavors in a refined contemporary setting. The injera is made fresh daily from fermented teff flour, providing the ideal tangy base for the richly spiced dishes.",
            "An Ethiopian coffee house and restaurant where the ancient coffee ceremony is practiced alongside hearty traditional meals. The strong, aromatic Ethiopian coffee is among the finest you'll find outside Addis Ababa.",
            "A casual Ethiopian spot known for its generous combination platters that allow diners to sample the full spectrum of Ethiopian cuisine. The slow-simmered stews and hand-torn injera make for deeply satisfying meals.",
        ]
    },
    "Brazilian": {
        "dishes": ["Feijoada", "Churrasco", "Pão de Queijo", "Açaí Bowl", "Coxinha",
                   "Moqueca", "Picanha", "Brigadeiro", "Caipirinha", "Pastel",
                   "Vatapá", "Acarajé", "Bobó de Camarão", "Farofa", "Romeo e Julieta"],
        "descriptions": [
            "A Brazilian churrascaria where gaucho-style grilled meats are carved tableside in endless succession. The caipirinha cocktails and vibrant salad bar complement the carnivore's paradise of perfectly seasoned cuts.",
            "A casual Brazilian café specializing in street food classics like coxinha, pão de queijo, and açaí bowls. The relaxed atmosphere and friendly service reflect the warmth of Brazilian culture.",
            "An upscale Brazilian restaurant showcasing the regional diversity of this vast nation's cuisine. From the seafood moqueca of Bahia to the hearty feijoada of Rio, the menu is a culinary journey across Brazil.",
            "A festive Brazilian restaurant and samba bar where the rhythms of Rio accompany a feast of grilled meats and tropical cocktails. Weekend performances and a lively dance floor create an unforgettable dining experience.",
            "A neighborhood Brazilian spot run by expats from São Paulo who bring an authentic taste of Brazil to their adopted home. The house-made sauces, fresh cassava dishes, and perfectly pulled caipirinha are the highlights.",
        ]
    },
    "Peruvian": {
        "dishes": ["Ceviche Clásico", "Lomo Saltado", "Aji de Gallina", "Causa Limeña",
                   "Anticuchos", "Cau Cau", "Papa a la Huancaína", "Arroz con Leche",
                   "Pisco Sour", "Tiradito", "Seco de Cordero", "Tacu Tacu", "Chicharrón de Calamar", "Arroz con Mariscos", "Suspiro Limeño"],
        "descriptions": [
            "A celebrated Peruvian restaurant reflecting the extraordinary biodiversity and multicultural influences that make Peruvian cuisine one of the world's great culinary traditions. The ceviche and tiradito selections showcase Peru's mastery of raw seafood.",
            "A modern Nikkei-Peruvian restaurant fusing the Japanese-Peruvian culinary tradition with innovative contemporary techniques. The results are startlingly original dishes that have earned the chef international recognition.",
            "A casual Peruvian criollo restaurant serving the hearty, comforting dishes of Lima's neighborhoods. The slow-braised stews, crispy anticuchos, and abundant sides reflect the soul of everyday Peruvian home cooking.",
            "An upscale Peruvian restaurant drawing inspiration from all regions of Peru, from the Pacific coast to the Amazonian jungle. The extensive pisco list and innovative cocktail program complement the adventurous menu beautifully.",
            "A neighborhood Peruvian spot where authenticity and generosity define the experience. The owner-chef from Arequipa brings regional specialties rarely found elsewhere, prepared with the warmth and pride of someone sharing their heritage.",
        ]
    },
    "Lebanese": {
        "dishes": ["Kibbeh Nayyeh", "Sfeeha", "Labneh", "Fattoush", "Hummus Beiruti",
                   "Kebbeh Bil Sayniyeh", "Kafta", "Tabbouleh", "Warak Dawali", "Knafeh",
                   "Shawarma", "Falafel", "Mujaddara", "Riz bi Shaeriyeh", "Maamoul"],
        "descriptions": [
            "An authentic Lebanese restaurant bringing the exuberant hospitality and extraordinary cuisine of Beirut to the table. The mezze tradition is fully honored here with dozens of small plates representing the diversity of Lebanese food.",
            "A modern Lebanese restaurant showcasing the bright, herb-forward flavors of Lebanese home cooking. The extensive vegetable and legume dishes make it a favorite among those seeking wholesome, plant-rich Mediterranean meals.",
            "A family-run Lebanese spot where grandmother's recipes for kibbeh, hummus, and kafta are prepared with love and shared generously. The warm, homelike atmosphere makes every guest feel like family.",
            "An upscale Lebanese restaurant presenting refined versions of classic dishes alongside innovative interpretations of Lebanese culinary heritage. The private dining rooms and extensive arak collection make it ideal for special occasions.",
            "A casual Lebanese café and bakery open all day for freshly baked pita, morning manousheh, and afternoon coffee with baklava. The simple, quality-driven menu celebrates Lebanon's extraordinary everyday food culture.",
        ]
    },
    "Turkish": {
        "dishes": ["Iskender Kebab", "Meze Platter", "Lahmacun", "Baklava", "Pide",
                   "Adana Kebab", "Mercimek Çorbası", "Köfte", "Manti", "Börek",
                   "Balık Ekmek", "Midye Dolma", "Kazandibi", "Sütlaç", "Raki"],
        "descriptions": [
            "A richly decorated Turkish restaurant evoking the grandeur of an Ottoman palace dining room. The extensive meze selection, slow-roasted meats, and exceptional Turkish wines celebrate the depth of this ancient culinary tradition.",
            "A casual Turkish pide and kebab shop producing arguably the finest Turkish flatbreads in the city. The wood-fired oven and hand-ground meat mixture are the secrets behind the extraordinary quality.",
            "A modern Anatolian restaurant presenting the regional diversity of Turkish cuisine that extends far beyond kebabs. The chef's journey through Turkey's culinary landscape results in a menu of surprising discovery and delight.",
            "A bustling Turkish restaurant and tea house where communal tables, strong çay, and generous portions create a convivial atmosphere. The breakfast menu featuring dozens of dishes is especially popular on weekends.",
            "A refined Turkish meyhane where raki, meze, and grilled seafood define the leisurely evening experience. The tradition of lingering over drinks and small bites late into the evening is fully embraced here.",
        ]
    },
    "Caribbean": {
        "dishes": ["Jerk Chicken", "Rice and Peas", "Oxtail Stew", "Ackee and Saltfish",
                   "Curry Goat", "Roti", "Plantains", "Conch Fritters", "Mango Chutney",
                   "Callaloo", "Rum Punch", "Doubles", "Pelau", "Breadfruit", "Bake and Shark"],
        "descriptions": [
            "A vibrant Caribbean restaurant where the warmth of the islands permeates everything from the jerk-spiced meats to the rum cocktails. The slow-smoked proteins and rice and peas reflect the soul of Jamaican cooking.",
            "An authentic Caribbean restaurant drawing from the culinary traditions of Jamaica, Trinidad, Barbados, and beyond. The curry goat, roti, and fresh seafood dishes represent the wonderful diversity of Caribbean island cuisines.",
            "A casual Caribbean street food spot bringing the flavors of Trinidad to the city with freshly made roti wraps, doubles, and pholourie. The pepper sauce is legendary and the portions are always generous.",
            "A festive Caribbean restaurant and rum bar where tropical cocktails, steel drum music, and island-inspired cuisine create a perpetual vacation atmosphere. The weekend brunch buffet is particularly celebrated.",
            "A family-owned Caribbean restaurant where dishes from grandmother's kitchen in Kingston inspire the entire menu. The slow-cooked oxtail, festival dumplings, and homemade ginger beer are the highlights of this beloved neighborhood spot.",
        ]
    }
}

RESTAURANT_NAME_TEMPLATES = {
    "Italian": ["La {adj} Cucina", "Ristorante {noun}", "Trattoria {noun}", "Il {adj} Forno", "Casa {noun}",
                "Osteria {noun}", "Da {noun}", "Al {adj} Tavolo", "Sotto {noun}", "Bella {noun}"],
    "Chinese": ["{noun} Garden", "Golden {noun}", "{noun} Palace", "Lucky {noun}", "Imperial {noun}",
                "{noun} House", "Red {noun}", "Jade {noun}", "Dragon {noun}", "Phoenix {noun}"],
    "Japanese": ["{noun} Sushi", "Sakura {noun}", "{noun} Ramen", "Hana {noun}", "Yoshi {noun}",
                 "{noun} Kitchen", "Mizu {noun}", "Tokyo {noun}", "{noun} Bar", "Nori {noun}"],
    "Mexican": ["El {adj} {noun}", "La {adj} {noun}", "Casa {noun}", "{noun} Grill", "Taqueria {noun}",
                "Los {noun}", "Las {noun}", "{noun} Cantina", "Hacienda {noun}", "Mi {adj} {noun}"],
    "Indian": ["{noun} Spice", "Royal {noun}", "{noun} Palace", "Taste of {noun}", "{noun} Kitchen",
               "Bombay {noun}", "Delhi {noun}", "{noun} Masala", "Taj {noun}", "Curry {noun}"],
    "Thai": ["Thai {noun}", "{noun} Bangkok", "Lotus {noun}", "{noun} Garden", "Golden {noun}",
             "Siam {noun}", "{noun} Palace", "Thai {adj} {noun}", "Orchid {noun}", "{noun} Kitchen"],
    "French": ["Le {adj} {noun}", "La {adj} {noun}", "Café {noun}", "Brasserie {noun}", "Bistro {noun}",
               "Maison {noun}", "Chez {noun}", "Le {noun}", "Les {noun}", "Au {adj} {noun}"],
    "American": ["The {adj} {noun}", "{noun} & Grill", "Big {noun}", "All-American {noun}", "{noun} House",
                 "The {noun}", "{noun} Kitchen", "Classic {noun}", "{noun} Bar & Grill", "Liberty {noun}"],
    "Mediterranean": ["{noun} Mediterranean", "Blue {noun}", "Olive {noun}", "Sea {noun}", "Coast {noun}",
                      "Aegean {noun}", "{noun} Mezze", "Santorini {noun}", "Mykonos {noun}", "{noun} Terrace"],
    "Korean": ["Seoul {noun}", "{noun} Korean BBQ", "Gangnam {noun}", "Korean {adj} {noun}", "K-{noun}",
               "{noun} Grill", "Hanok {noun}", "Kimchi {noun}", "{noun} House", "Bap {noun}"],
    "Vietnamese": ["Pho {noun}", "{noun} Vietnamese", "Saigon {noun}", "Hanoi {noun}", "Mekong {noun}",
                   "{noun} Garden", "Lotus {noun}", "Pho {adj} {noun}", "{noun} Kitchen", "Bao {noun}"],
    "Greek": ["Santorini {noun}", "Mykonos {noun}", "Athena {noun}", "Greek {noun}", "Olympus {noun}",
              "{noun} Taverna", "Acropolis {noun}", "Zeus {noun}", "{noun} Greek", "Hellas {noun}"],
    "Spanish": ["El {noun}", "La {noun}", "Casa {noun}", "Tapas {noun}", "{noun} Spanish",
                "Madrid {noun}", "Barcelona {noun}", "{noun} Tapas", "Iberia {noun}", "Sol {noun}"],
    "Middle Eastern": ["{noun} Mediterranean", "Al {noun}", "Beirut {noun}", "Arabian {noun}", "{noun} Kitchen",
                       "Oasis {noun}", "Desert {noun}", "{noun} Grill", "Levant {noun}", "Silk Road {noun}"],
    "Ethiopian": ["Addis {noun}", "Haile {noun}", "Ethiopian {noun}", "{noun} Ethiopian", "Nile {noun}",
                  "Abyssinia {noun}", "Axum {noun}", "Lalibela {noun}", "{noun} Kitchen", "Injera {noun}"],
    "Brazilian": ["Rio {noun}", "Brazil {noun}", "Copacabana {noun}", "Amazonia {noun}", "{noun} Churrasco",
                  "Samba {noun}", "Ipanema {noun}", "{noun} Brazilian", "Tropical {noun}", "Verde {noun}"],
    "Peruvian": ["Lima {noun}", "Machu Picchu {noun}", "Peru {noun}", "Andean {noun}", "{noun} Peruvian",
                 "Ceviche {noun}", "Pachamama {noun}", "{noun} Lima", "Inca {noun}", "Coastal {noun}"],
    "Lebanese": ["Beirut {noun}", "Cedar {noun}", "Lebanese {noun}", "{noun} Lebanese", "Levant {noun}",
                 "Phoenicia {noun}", "{noun} Mezze", "Byblos {noun}", "Baalbek {noun}", "Zaarour {noun}"],
    "Turkish": ["Istanbul {noun}", "Ottoman {noun}", "Turkish {noun}", "{noun} Turkish", "Bosphorus {noun}",
                "Anatolia {noun}", "Topkapi {noun}", "{noun} Kebab", "Sultan {noun}", "Ankara {noun}"],
    "Caribbean": ["Island {noun}", "Carib {noun}", "Tropical {noun}", "Caribbean {noun}", "Reggae {noun}",
                  "Jerk {noun}", "Rum {noun}", "Coconut {noun}", "{noun} Island", "Monsoon {noun}"]
}

NAME_NOUNS = ["Kitchen", "Bistro", "Table", "Corner", "Place", "Spot", "Eatery", "House", "Garden", "Café",
              "Terrace", "Bar", "Grill", "Room", "Lounge", "Nook", "Quarter", "Hub", "Haven", "Den"]
NAME_ADJS = ["Blue", "Golden", "Little", "Grand", "Classic", "Modern", "Original", "Fine", "True", "Fine"]

def generate_name(cuisine):
    templates = RESTAURANT_NAME_TEMPLATES.get(cuisine, ["{noun} Restaurant"])
    template = random.choice(templates)
    name = template.replace("{noun}", random.choice(NAME_NOUNS))
    name = name.replace("{adj}", random.choice(NAME_ADJS))
    return name

def generate_restaurants(n=300):
    restaurants = []
    used_names = set()

    # Distribute evenly across cuisines
    per_cuisine = n // len(CUISINES)
    extras = n % len(CUISINES)

    id_counter = 1
    for i, cuisine in enumerate(CUISINES):
        count = per_cuisine + (1 if i < extras else 0)
        cuisine_data = CUISINE_DATA[cuisine]

        for j in range(count):
            # Generate unique name
            for _ in range(20):
                name = generate_name(cuisine)
                if name not in used_names:
                    used_names.add(name)
                    break

            price_weights = [0.2, 0.4, 0.3, 0.1]
            price_range = random.choices(PRICE_RANGES, weights=price_weights)[0]

            # Rating: weighted toward 3.5-4.8
            rating = round(random.triangular(2.5, 5.0, 4.2), 1)

            # Pick 1-3 dietary options
            n_dietary = random.randint(0, 3)
            dietary_opts = random.sample(DIETARY, n_dietary)

            # Pick 3-6 features
            n_features = random.randint(3, 6)
            features = random.sample(FEATURES, n_features)

            # Pick 4-6 popular dishes
            n_dishes = random.randint(4, 6)
            popular_dishes = random.sample(cuisine_data["dishes"], min(n_dishes, len(cuisine_data["dishes"])))

            # Description
            description = random.choice(cuisine_data["descriptions"])

            ambiance = random.choice(AMBIANCES)
            location = random.choice(LOCATIONS)

            # Generate review count correlated with rating
            review_count = int(random.randint(15, 2500))

            restaurant = {
                "id": id_counter,
                "name": name,
                "description": description,
                "cuisine": cuisine,
                "price_range": price_range,
                "rating": rating,
                "review_count": review_count,
                "ambiance": ambiance,
                "location": location,
                "dietary_options": dietary_opts,
                "popular_dishes": popular_dishes,
                "features": features,
                "tags": [cuisine.lower(), ambiance.lower(), price_range, location.lower().replace(" ", "-")]
            }
            restaurants.append(restaurant)
            id_counter += 1

    random.shuffle(restaurants)
    # Reassign IDs after shuffle
    for idx, r in enumerate(restaurants):
        r["id"] = idx + 1

    return restaurants

if __name__ == "__main__":
    restaurants = generate_restaurants(300)
    output_path = "data/restaurants.json"
    with open(output_path, "w") as f:
        json.dump(restaurants, f, indent=2)
    print(f"Generated {len(restaurants)} restaurants → {output_path}")

    # Print cuisine distribution
    from collections import Counter
    cuisines = Counter(r["cuisine"] for r in restaurants)
    for cuisine, count in sorted(cuisines.items()):
        print(f"  {cuisine}: {count}")
