import numpy as np
import matplotlib.pyplot as plt

train_1 = [0.11069708063907266, 0.08811409276961697, 0.08105955980354101, 0.07558082122998828, 0.07011950069413511, 0.06691113400731102, 0.06260602588155564, 0.05935029934861179, 0.05723769738906175, 0.05555064214775733, 0.05335173866308085, 0.05129444468705779, 0.050478895701443725, 0.04931945561853022, 0.04715523586065451, 0.04605491985400065, 0.0451838770632538, 0.04334549944398846, 0.04248872802802134, 0.04197140375959174, 0.0407277213603695, 0.03933370954900106, 0.03873051027084795, 0.03767121634794951, 0.037104131053247356, 0.035888998811694425, 0.03562909381081887, 0.03504841744365832, 0.03434123497771809, 0.03357712481874694, 0.03343084371111777, 0.03221100467144958, 0.032077544177685186, 0.0314505403066594, 0.030923189737922012, 0.030463831396075725, 0.0302111609999685, 0.0295227153881996, 0.029204114055085065, 0.028700388525081767, 0.028308890535456738, 0.0276919654707033, 0.02756880201605613, 0.02723218290687384, 0.027356997165671196, 0.026650992892561207, 0.02671283975255703, 0.02631189630454837, 0.026017916000505806, 0.025341726903248492, 0.025134418284101866, 0.02497976263287304, 0.024864422403130353, 0.024460197975464674, 0.02437672446745809, 0.02424442538148223, 0.023652410492640753, 0.023620967662222804, 0.023447901162194508, 0.023144897024969326]
val_1 = [0.06305601416373408, 0.058795657758782434, 0.05227777978719829, 0.045650300049743096, 0.042554267887751775, 0.03974774779824467, 0.0378010683729277, 0.03614966239925329, 0.03252258795228871, 0.03128947075698283, 0.02986900589440937, 0.02921753740784797, 0.028475670893858007, 0.027485110917938996, 0.026383990192084344, 0.025917099764594782, 0.02443722162589237, 0.024370732185031687, 0.02329876809683326, 0.022769094493191738, 0.02218272304767138, 0.02186829225438369, 0.020752493558185443, 0.020909333710449857, 0.020158477853257934, 0.01966681852694843, 0.01911365878223986, 0.01897509673053955, 0.018323773167446836, 0.017974349875728806, 0.01764909452577303, 0.017180815254422752, 0.0172924499886183, 0.01713719235902483, 0.01658962777053768, 0.0162795235390787, 0.016051398771633574, 0.016166397134543625, 0.015603007803150973, 0.015543886351507979, 0.015219180766106039, 0.015209040840386183, 0.014923548307753616, 0.014723085858798647, 0.014490790332534484, 0.014611744575879791, 0.014426264071280693, 0.0141730834669494, 0.013909118032300627, 0.013854617500895417, 0.014104799542866357, 0.013823026942403673, 0.013834865025982454, 0.013864451662528438, 0.013539735584628659, 0.01341960651122711, 0.013181612379365153, 0.013109527721807554, 0.013085095961759616, 0.013126748541442605]
train_2 = [0.1118167996867666, 0.08771446575365159, 0.08160864001182469, 0.074970735643508, 0.06972639853019473, 0.0651034152723776, 0.06266860170016848, 0.05921611167393406, 0.05626244691324933, 0.054578260740065034, 0.05250453132323023, 0.051159626473627186, 0.049888700076927965, 0.04837053759381515, 0.04683521048375178, 0.045512518878868424, 0.04455541113106358, 0.04269776021582393, 0.04178417599623483, 0.04056263526944073, 0.03960242882637908, 0.03891973718636781, 0.03830401934926013, 0.03695906800633533, 0.03631850045344243, 0.03570329559608954, 0.03476294718609959, 0.034310405631018774, 0.03348958545064111, 0.03289483881673809, 0.03246333050854043, 0.03181207325333784, 0.031550787977801475, 0.030900635510828674, 0.030414512584210414, 0.029914985398734627, 0.02919867387229534, 0.028843477541271367, 0.028671591075483092, 0.028217746933819027, 0.027913539445919594, 0.027731489732934326, 0.02714151045131567, 0.026545843810255636, 0.026121458414857086, 0.02615023296969527, 0.026017471926487037, 0.025475308016937603, 0.025135603906277337, 0.025193971414907746, 0.02486154623633002, 0.02411732644279539, 0.024401515285432922, 0.023931939531032738, 0.023822589689434934, 0.02347388852035766, 0.02345102144168623, 0.0227311204615313, 0.022852282209704092, 0.022322128308445505]
val_2 = [0.06305601416373408, 0.058795657758782434, 0.05227777978719829, 0.045650300049743096, 0.042554267887751775, 0.03974774779824467, 0.0378010683729277, 0.03614966239925329, 0.03252258795228871, 0.03128947075698283, 0.02986900589440937, 0.02921753740784797, 0.028475670893858007, 0.027485110917938996, 0.026383990192084344, 0.025917099764594782, 0.02443722162589237, 0.024370732185031687, 0.02329876809683326, 0.022769094493191738, 0.02218272304767138, 0.02186829225438369, 0.020752493558185443, 0.020909333710449857, 0.020158477853257934, 0.01966681852694843, 0.01911365878223986, 0.01897509673053955, 0.018323773167446836, 0.017974349875728806, 0.01764909452577303, 0.017180815254422752, 0.0172924499886183, 0.01713719235902483, 0.01658962777053768, 0.0162795235390787, 0.016051398771633574, 0.016166397134543625, 0.015603007803150973, 0.015543886351507979, 0.015219180766106039, 0.015209040840386183, 0.014923548307753616, 0.014723085858798647, 0.014490790332534484, 0.014611744575879791, 0.014426264071280693, 0.0141730834669494, 0.013909118032300627, 0.013854617500895417, 0.014104799542866357, 0.013823026942403673, 0.013834865025982454, 0.013864451662528438, 0.013539735584628659, 0.01341960651122711, 0.013181612379365153, 0.013109527721807554, 0.013085095961759616, 0.013126748541442605]
train_3 = [0.10849124286627149, 0.08767257840751826, 0.08105718058075889, 0.07452966327436196, 0.06982369991535084, 0.06571087340675659, 0.061658743052995166, 0.05852701705470924, 0.05656733255263066, 0.054535528638852145, 0.052832839179310816, 0.0514064799472447, 0.04941771143034149, 0.048785339494869844, 0.04698210548160519, 0.04568717544382675, 0.0442832674685538, 0.043089546660622674, 0.04237980208684256, 0.041276341965513046, 0.04002417525100591, 0.0395954099911529, 0.03815999992968399, 0.037408477520564094, 0.03659709340950371, 0.0360472084930112, 0.034825531850760066, 0.034632080750303273, 0.03318087910438496, 0.033022840382949926, 0.032381106604873554, 0.031859256139067565, 0.03135818405932343, 0.030447928911432963, 0.030048002055102916, 0.029871425251643317, 0.02927733953520413, 0.02888990671097844, 0.028537985808079522, 0.0279846821580968, 0.02767441884374871, 0.02734364206669094, 0.027243950892640246, 0.026568767827016327, 0.026452403840175085, 0.026196448390253785, 0.025529665029107165, 0.02574866542331753, 0.025064826457600446, 0.024786427486627228, 0.024820425471818796, 0.02419574886366288, 0.024011665718563217, 0.023703800948148637, 0.023935316358040713, 0.02363777192800946, 0.022976797899225247, 0.02317635934338799, 0.02293511038700707, 0.022675755157678446]
val_3 = [0.06305601416373408, 0.058795657758782434, 0.05227777978719829, 0.045650300049743096, 0.042554267887751775, 0.03974774779824467, 0.0378010683729277, 0.03614966239925329, 0.03252258795228871, 0.03128947075698283, 0.02986900589440937, 0.02921753740784797, 0.028475670893858007, 0.027485110917938996, 0.026383990192084344, 0.025917099764594782, 0.02443722162589237, 0.024370732185031687, 0.02329876809683326, 0.022769094493191738, 0.02218272304767138, 0.02186829225438369, 0.020752493558185443, 0.020909333710449857, 0.020158477853257934, 0.01966681852694843, 0.01911365878223986, 0.01897509673053955, 0.018323773167446836, 0.017974349875728806, 0.01764909452577303, 0.017180815254422752, 0.0172924499886183, 0.01713719235902483, 0.01658962777053768, 0.0162795235390787, 0.016051398771633574, 0.016166397134543625, 0.015603007803150973, 0.015543886351507979, 0.015219180766106039, 0.015209040840386183, 0.014923548307753616, 0.014723085858798647, 0.014490790332534484, 0.014611744575879791, 0.014426264071280693, 0.0141730834669494, 0.013909118032300627, 0.013854617500895417, 0.014104799542866357, 0.013823026942403673, 0.013834865025982454, 0.013864451662528438, 0.013539735584628659, 0.01341960651122711, 0.013181612379365153, 0.013109527721807554, 0.013085095961759616, 0.013126748541442605]

# Average the training arrays and validation arrays
train_avg = np.mean([train_1, train_2, train_3], axis=0)
val_avg = np.mean([val_1, val_2, val_3], axis=0)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(train_avg, label='Training Average')
plt.plot(val_avg, label='Validation Average')
plt.title('Federated Clients Average Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Average Value')
plt.legend()
plt.show()