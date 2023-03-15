################################################Ignore below here unless you like looking at data structures #########################
def get_example_drug(N=5):

    if N == 4:
        drug = {'0000': 0.46217992461006,
                '0001': 0.8155712092099533,
                '0010': 0.7950926864250307,
                '0011': 0.0,
                '0100': 0.4999713293529357,
                '0101': 0.9593406497722251,
                '0110': 0.9461778059648714,
                '0111': 0.17416994306275166,
                '1000': 0.9755636165011852,
                '1001': 0.9009413679468488,
                '1010': 0.6506415682852436,
                '1011': 0.4850063904158733,
                '1100': 0.33842785595107555,
                '1101': 0.36376400420426774,
                '1110': 1.0,
                '1111': 0.9044483660888123}
    elif N==5:
        drug = {'00000': 0.3717050099762647,
                '00001': 0.0,
                '00010': 0.3776538234165811,
                '00011': 0.11356674994711251,
                '00100': 0.39100242325177986,
                '00101': 0.14495002138002858,
                '00110': 0.8358585152487236,
                '00111': 0.4528591854802891,
                '01000': 0.35937170496384463,
                '01001': 0.14651193489659822,
                '01010': 0.6585318605281533,
                '01011': 0.3325700008963652,
                '01100': 0.6567222215152005,
                '01101': 0.4381547489450053,
                '01110': 0.6124949006296058,
                '01111': 0.4787110317929099,
                '10000': 0.6114187992210784,
                '10001': 0.14090040244794025,
                '10010': 0.8144935654992457,
                '10011': 0.3030002921808745,
                '10100': 0.7202963454312571,
                '10101': 0.8151430016503007,
                '10110': 0.8215791978264813,
                '10111': 0.9392714136367594,
                '11000': 0.539040393512979,
                '11001': 0.44612832485096,
                '11010': 0.4707569659419544,
                '11011': 0.546077521783412,
                '11100': 0.7649481766258261,
                '11101': 0.7205640683494234,
                '11110': 0.9368918255692954,
                '11111': 1.0}

    elif N==6: 
        drug = {'000000': 0.25946663889100047,
                '000001': 0.556085476497859,
                '000010': 0.46689997187807813,
                '000011': 0.3294633327358319,
                '000100': 0.4031935307549663,
                '000101': 0.2947094359828658,
                '000110': 0.5534374802242279,
                '000111': 0.5495150143404818,
                '001000': 0.3668687223763278,
                '001001': 0.44339907159231856,
                '001010': 0.603513795195309,
                '001011': 0.5854450022909445,
                '001100': 0.6788355999275285,
                '001101': 0.5118212743532856,
                '001110': 0.6510941591565972,
                '001111': 1.0,
                '010000': 0.26194368907639365,
                '010001': 0.12410655708067952,
                '010010': 0.43976338744263144,
                '010011': 0.4162917287295355,
                '010100': 0.43482284360989587,
                '010101': 0.12173833900653652,
                '010110': 0.31000739117328346,
                '010111': 0.44137561876610887,
                '011000': 0.42316069487225677,
                '011001': 0.009034675958542413,
                '011010': 0.7001871783762964,
                '011011': 0.5982331271106044,
                '011100': 0.38433729376597137,
                '011101': 0.3578419282562447,
                '011110': 0.2667466946401551,
                '011111': 0.5967974521826113,
                '100000': 0.498064226822892,
                '100001': 0.12810805296653,
                '100010': 0.26322327511027904,
                '100011': 0.5735867712251981,
                '100100': 0.10846088541218113,
                '100101': 0.2051993361538113,
                '100110': 0.29585221422105024,
                '100111': 0.3858385921755399,
                '101000': 0.19576634083979041,
                '101001': 0.0,
                '101010': 0.635167879924397,
                '101011': 0.28957922347745235,
                '101100': 0.4086623830331319,
                '101101': 0.38444354157190846,
                '101110': 0.5349397565504158,
                '101111': 0.37827587810994073,
                '110000': 0.10998268624092429,
                '110001': 0.3680992006061166,
                '110010': 0.22928989072656483,
                '110011': 0.4301173187427265,
                '110100': 0.3152388973719659,
                '110101': 0.10550512513352689,
                '110110': 0.6889777322147395,
                '110111': 0.5053390487934716,
                '111000': 0.2779081398067359,
                '111001': 0.16114930206487532,
                '111010': 0.4158207518137712,
                '111011': 0.1951452372463429,
                '111100': 0.17984518972775743,
                '111101': 0.28249471025699957,
                '111110': 0.33562790856784863,
                '111111': 0.08530892933813033}
    elif N==7: 
        drug = {'0000000': 0.6562282138811357,
        '0000001': 0.6962873453296857,
        '0000010': 0.15111717533915714,
        '0000011': 0.5036349151344484,
        '0000100': 0.2218358418731532,
        '0000101': 0.5324488144174444,
        '0000110': 0.32522021353363656,
        '0000111': 0.8551117100717627,
        '0001000': 0.3072333499105162,
        '0001001': 0.6068314227050945,
        '0001010': 0.8334969749713582,
        '0001011': 0.44185208556347244,
        '0001100': 0.4768863145351982,
        '0001101': 0.4505953533685471,
        '0001110': 0.24610705916371164,
        '0001111': 0.2371775021499528,
        '0010000': 0.6960736600587111,
        '0010001': 0.7166408183604924,
        '0010010': 0.381198724178471,
        '0010011': 0.6151663310832958,
        '0010100': 0.44939091320630803,
        '0010101': 0.227030995671083,
        '0010110': 0.5271270362189533,
        '0010111': 0.5063666033922998,
        '0011000': 0.38321850769941884,
        '0011001': 0.3593675170241291,
        '0011010': 0.6119789132728971,
        '0011011': 0.15199854278608016,
        '0011100': 0.3173213955431792,
        '0011101': 0.5584867108627407,
        '0011110': 0.19861408059061583,
        '0011111': 0.644578174435678,
        '0100000': 0.6709726301273546,
        '0100001': 0.6368207057848099,
        '0100010': 0.6188615608115964,
        '0100011': 0.48186578895617416,
        '0100100': 0.5798858850317224,
        '0100101': 0.3716823381414905,
        '0100110': 0.7476338815155691,
        '0100111': 0.4638941470518578,
        '0101000': 0.8837676664234708,
        '0101001': 0.4150049827045108,
        '0101010': 0.4771014102620448,
        '0101011': 0.4916900903509899,
        '0101100': 0.5749994076687622,
        '0101101': 0.5623931340912197,
        '0101110': 0.6945627415031554,
        '0101111': 0.6032827114723909,
        '0110000': 0.6924498139083567,
        '0110001': 0.46435976236490706,
        '0110010': 0.9925137993477746,
        '0110011': 1.0,
        '0110100': 0.5030692950697736,
        '0110101': 0.6657268269301292,
        '0110110': 0.3326493694666408,
        '0110111': 0.4539725054898229,
        '0111000': 0.5127518168665205,
        '0111001': 0.41395077663977703,
        '0111010': 0.522040770388562,
        '0111011': 0.5930008172913007,
        '0111100': 0.4598164848356433,
        '0111101': 0.7207598199287827,
        '0111110': 0.4449905035300315,
        '0111111': 0.7127167586277421,
        '1000000': 0.2795343994967053,
        '1000001': 0.9413754473564785,
        '1000010': 0.5565369387865349,
        '1000011': 0.8311849312864616,
        '1000100': 0.44211792119487453,
        '1000101': 0.4463261309714834,
        '1000110': 0.45162724573161883,
        '1000111': 0.5420919715873942,
        '1001000': 0.3223711380775945,
        '1001001': 0.4651067143250024,
        '1001010': 0.5258731084620836,
        '1001011': 0.5638831971504414,
        '1001100': 0.5401992260882207,
        '1001101': 0.45602023768146016,
        '1001110': 0.4484771721980634,
        '1001111': 0.14295359472893387,
        '1010000': 0.5517673557214535,
        '1010001': 0.8129701838106612,
        '1010010': 0.44510449678140174,
        '1010011': 0.37382316011887773,
        '1010100': 0.46947132930509855,
        '1010101': 0.10661039217937039,
        '1010110': 0.5827448072968298,
        '1010111': 0.25895131427577983,
        '1011000': 0.11928669128588537,
        '1011001': 0.387762120099263,
        '1011010': 0.6769061389442661,
        '1011011': 0.6786019517018235,
        '1011100': 0.3174446713615433,
        '1011101': 0.16456933699551723,
        '1011110': 0.5126468985704199,
        '1011111': 0.8756987683116964,
        '1100000': 0.43456116860555005,
        '1100001': 0.6364847320027166,
        '1100010': 0.2915231158672259,
        '1100011': 0.6802557550659352,
        '1100100': 0.5903415322149353,
        '1100101': 0.6279197615432534,
        '1100110': 0.8252526660033257,
        '1100111': 0.42781776917125924,
        '1101000': 0.6312660446884152,
        '1101001': 0.44553175485014995,
        '1101010': 0.6465752383729088,
        '1101011': 0.7240712616042392,
        '1101100': 0.49091728341448465,
        '1101101': 0.44086622365489747,
        '1101110': 0.12379725007187876,
        '1101111': 0.5001733829758883,
        '1110000': 0.8000245290294943,
        '1110001': 0.0,
        '1110010': 0.5076177411289564,
        '1110011': 0.4459107677727714,
        '1110100': 0.5609550185827152,
        '1110101': 0.5993932749361348,
        '1110110': 0.5559437566839036,
        '1110111': 0.30926411191860265,
        '1111000': 0.37839518001562755,
        '1111001': 0.5763086019273285,
        '1111010': 0.6266393567544578,
        '1111011': 0.45309019862645866,
        '1111100': 0.6185072489981226,
        '1111101': 0.433187260348499,
        '1111110': 0.4070624654229378,
        '1111111': 0.1763369792371006}
    elif N==8:
        drug = {'00000000': 0.5420079432590408,
                '00000001': 0.791763848316879,
                '00000010': 0.6881989397253676,
                '00000011': 0.7843535423588631,
                '00000100': 0.5059651958128084,
                '00000101': 0.8884286548694686,
                '00000110': 0.5104252228025047,
                '00000111': 0.5121841362841011,
                '00001000': 0.39664706102360486,
                '00001001': 0.7225370605680946,
                '00001010': 0.2662663022030919,
                '00001011': 0.6517855379720546,
                '00001100': 0.43725004486897756,
                '00001101': 0.7261927782586104,
                '00001110': 0.7800750935469452,
                '00001111': 0.8164606729970897,
                '00010000': 0.1858234353818361,
                '00010001': 0.4564293361895631,
                '00010010': 0.47334146500224655,
                '00010011': 0.8729227503218333,
                '00010100': 0.47449443361189075,
                '00010101': 0.6801460328245728,
                '00010110': 0.6062256701015828,
                '00010111': 0.7400907756605766,
                '00011000': 0.301888523871341,
                '00011001': 0.8267346015955715,
                '00011010': 0.5241221143856023,
                '00011011': 0.88728742522356,
                '00011100': 0.655369975862686,
                '00011101': 0.36051393572424995,
                '00011110': 0.5201916087580353,
                '00011111': 0.590302119265284,
                '00100000': 0.5391031791754356,
                '00100001': 0.7084803264815833,
                '00100010': 0.3853675605907982,
                '00100011': 0.5774389738297363,
                '00100100': 0.7289684451375523,
                '00100101': 0.714114091204817,
                '00100110': 0.48827428269824474,
                '00100111': 1.0,
                '00101000': 0.6517858736529206,
                '00101001': 0.594188352789435,
                '00101010': 0.331884521451309,
                '00101011': 0.3105382850408996,
                '00101100': 0.3981806015376312,
                '00101101': 0.662460000718499,
                '00101110': 0.45209425824646404,
                '00101111': 0.7505801340917013,
                '00110000': 0.46875843908862475,
                '00110001': 0.617302931557071,
                '00110010': 0.4325000910907778,
                '00110011': 0.46212232968448697,
                '00110100': 0.41072643196141295,
                '00110101': 0.347506163171187,
                '00110110': 0.45028980931630547,
                '00110111': 0.5219732149996048,
                '00111000': 0.25819971964232824,
                '00111001': 0.5828477069195485,
                '00111010': 0.5727063062137673,
                '00111011': 0.5614200805610501,
                '00111100': 0.4033173483389983,
                '00111101': 0.4509399371226897,
                '00111110': 0.40023866053302826,
                '00111111': 0.7310851221460976,
                '01000000': 0.5623952443058895,
                '01000001': 0.4002043547655842,
                '01000010': 0.37446150198077915,
                '01000011': 0.6797760288429092,
                '01000100': 0.6859220261685194,
                '01000101': 0.4930454623928371,
                '01000110': 0.6998275918286744,
                '01000111': 0.5224924101496355,
                '01001000': 0.3338894306918084,
                '01001001': 0.5599109036514305,
                '01001010': 0.541198621082275,
                '01001011': 0.5773404022646019,
                '01001100': 0.45591620293613616,
                '01001101': 0.9053089519432963,
                '01001110': 0.45310483113087274,
                '01001111': 0.7630509749132784,
                '01010000': 0.5319812162337986,
                '01010001': 0.608105325510297,
                '01010010': 0.5584460219480009,
                '01010011': 0.884023273106652,
                '01010100': 0.4883992295742217,
                '01010101': 0.6255959032566227,
                '01010110': 0.5434981133345779,
                '01010111': 0.577371038142283,
                '01011000': 0.37660943049502976,
                '01011001': 0.49383504003030254,
                '01011010': 0.39836788839664555,
                '01011011': 0.380185793697769,
                '01011100': 0.4137959020960565,
                '01011101': 0.4445023871867444,
                '01011110': 0.3780325500386527,
                '01011111': 0.807038820120527,
                '01100000': 0.723284896162292,
                '01100001': 0.3973563010786981,
                '01100010': 0.33242819273071844,
                '01100011': 0.6313475453763825,
                '01100100': 0.4132279222452509,
                '01100101': 0.6373705519844814,
                '01100110': 0.5838389941138464,
                '01100111': 0.4695642525154337,
                '01101000': 0.42943617782098814,
                '01101001': 0.42962178660777617,
                '01101010': 0.38442422142772076,
                '01101011': 0.660150551777247,
                '01101100': 0.4844059284919529,
                '01101101': 0.46913014019633154,
                '01101110': 0.2910960957163692,
                '01101111': 0.7784261220586629,
                '01110000': 0.0,
                '01110001': 0.5887485579781069,
                '01110010': 0.20076838471043215,
                '01110011': 0.5133894577049473,
                '01110100': 0.7831633299494036,
                '01110101': 0.722279086932303,
                '01110110': 0.2390623249872252,
                '01110111': 0.7530648996789936,
                '01111000': 0.495279355709113,
                '01111001': 0.2947065363971685,
                '01111010': 0.10856277568083503,
                '01111011': 0.3207472753442778,
                '01111100': 0.4575319177127637,
                '01111101': 0.5600398131992629,
                '01111110': 0.4918610543378061,
                '01111111': 0.7078248345801368,
                '10000000': 0.4094814280192285,
                '10000001': 0.6037950730752683,
                '10000010': 0.5607397580498803,
                '10000011': 0.8140901459487246,
                '10000100': 0.4673608097126418,
                '10000101': 0.382028608777037,
                '10000110': 0.5498387573890496,
                '10000111': 0.6615229531062353,
                '10001000': 0.1797272892348437,
                '10001001': 0.4760455783178197,
                '10001010': 0.6013564202859022,
                '10001011': 0.4759593337119254,
                '10001100': 0.42449801432954676,
                '10001101': 0.6674313741824544,
                '10001110': 0.6538701924735174,
                '10001111': 0.7103024342792944,
                '10010000': 0.6869632813922749,
                '10010001': 0.4929570605722377,
                '10010010': 0.7150685805130661,
                '10010011': 0.5959449644853959,
                '10010100': 0.5457814799206065,
                '10010101': 0.8032795263198085,
                '10010110': 0.5542323152728157,
                '10010111': 0.696242855459361,
                '10011000': 0.5349210794724603,
                '10011001': 0.8584557211095324,
                '10011010': 0.5514738191934804,
                '10011011': 0.5767081869961099,
                '10011100': 0.3005995516051469,
                '10011101': 0.6417876047041647,
                '10011110': 0.5184140558876684,
                '10011111': 0.6492770613296267,
                '10100000': 0.41745133489552844,
                '10100001': 0.4422794219680844,
                '10100010': 0.3744382525947805,
                '10100011': 0.7705931258851655,
                '10100100': 0.5013639580131366,
                '10100101': 0.6838579461573285,
                '10100110': 0.6748085380584257,
                '10100111': 0.9018136260530962,
                '10101000': 0.43968964266415506,
                '10101001': 0.5876665889211005,
                '10101010': 0.17868289473417828,
                '10101011': 0.7206598882542027,
                '10101100': 0.5667659650718399,
                '10101101': 0.7264084953973484,
                '10101110': 0.8234665522615551,
                '10101111': 0.8627766272629791,
                '10110000': 0.12636991344910042,
                '10110001': 0.48560232785855945,
                '10110010': 0.35523272951656937,
                '10110011': 0.5413726898328333,
                '10110100': 0.24771618024862116,
                '10110101': 0.580988880585363,
                '10110110': 0.49027808052081984,
                '10110111': 0.8080877011914953,
                '10111000': 0.39171281002707375,
                '10111001': 0.3147053509081689,
                '10111010': 0.3699138503200438,
                '10111011': 0.5218048358332429,
                '10111100': 0.23529827284766935,
                '10111101': 0.39723855715068224,
                '10111110': 0.6330416935108001,
                '10111111': 0.7508095245885702,
                '11000000': 0.5017134690963514,
                '11000001': 0.6223834197432814,
                '11000010': 0.2552303756731989,
                '11000011': 0.3878283437409375,
                '11000100': 0.37339302846613376,
                '11000101': 0.6093386192761006,
                '11000110': 0.02551079929217104,
                '11000111': 0.719431978147835,
                '11001000': 0.30160149322843444,
                '11001001': 0.17414305528985463,
                '11001010': 0.15373896557798672,
                '11001011': 0.19007155682533264,
                '11001100': 0.5135924671763386,
                '11001101': 0.8603345737617835,
                '11001110': 0.5776953556978374,
                '11001111': 0.7109738190265366,
                '11010000': 0.4493407528007171,
                '11010001': 0.567566428156925,
                '11010010': 0.2943045199126101,
                '11010011': 0.3374938018978211,
                '11010100': 0.4616714232546647,
                '11010101': 0.5136974867006299,
                '11010110': 0.3928285768264729,
                '11010111': 0.5155849641834649,
                '11011000': 0.25852035100424176,
                '11011001': 0.6730223778289729,
                '11011010': 0.5923509070334894,
                '11011011': 0.5201137424204667,
                '11011100': 0.1721491695808981,
                '11011101': 0.7776887459617641,
                '11011110': 0.15674784167795938,
                '11011111': 0.5830317677903332,
                '11100000': 0.6324395868847501,
                '11100001': 0.5146625430431537,
                '11100010': 0.38182937360016245,
                '11100011': 0.5339447806137023,
                '11100100': 0.529857326486236,
                '11100101': 0.5700748381050873,
                '11100110': 0.4051333759286761,
                '11100111': 0.629946117404353,
                '11101000': 0.5886896889339072,
                '11101001': 0.25481294049368314,
                '11101010': 0.392045924873265,
                '11101011': 0.633748677806512,
                '11101100': 0.4909576492333938,
                '11101101': 0.5257519168096054,
                '11101110': 0.3020753779217406,
                '11101111': 0.5293659131841024,
                '11110000': 0.34038802027281334,
                '11110001': 0.641093591324079,
                '11110010': 0.20462727269293673,
                '11110011': 0.5915487679617368,
                '11110100': 0.257688159218322,
                '11110101': 0.2663125112004504,
                '11110110': 0.4327340570829218,
                '11110111': 0.5757879134867641,
                '11111000': 0.10582926698814354,
                '11111001': 0.6126263351741214,
                '11111010': 0.2207806802473512,
                '11111011': 0.34368312371000065,
                '11111100': 0.47404311627841095,
                '11111101': 0.1316760051419533,
                '11111110': 0.4236513219733742,
                '11111111': 0.25722231302310317}
        
    return drug
