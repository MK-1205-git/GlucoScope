"""Survey coding. The binary target combines prediabetes and diabetes per UCI.
Age, education, and income are ordered survey codes, not literal quantities.
"""
FEATURES = [
 ('HighBP','High blood pressure',0,1,'Ever told you had high blood pressure?'),
 ('HighChol','High cholesterol',0,1,'Ever told you had high cholesterol?'),
 ('CholCheck','Cholesterol checked',0,1,'Cholesterol check within the past five years?'),
 ('BMI','Body mass index',12,98,'Integer BMI as encoded in this dataset; unusual values warrant caution.'),
 ('Smoker','Smoking history',0,1,'At least 100 cigarettes in your lifetime, not necessarily current smoking.'),
 ('Stroke','Stroke history',0,1,'Ever told you had a stroke?'),
 ('HeartDiseaseorAttack','Heart disease / heart attack',0,1,'Reported coronary heart disease or myocardial infarction.'),
 ('PhysActivity','Physical activity',0,1,'Physical activity in the past 30 days, excluding work.'),
 ('Fruits','Fruit daily',0,1,'Eat fruit at least once per day?'),
 ('Veggies','Vegetables daily',0,1,'Eat vegetables at least once per day?'),
 ('HvyAlcoholConsump','Heavy alcohol consumption',0,1,'Dataset definition: over 14 drinks/week for men or over 7 for women.'),
 ('AnyHealthcare','Healthcare coverage',0,1,'Any form of healthcare coverage?'),
 ('NoDocbcCost','Cost prevented care',0,1,'Needed a doctor in the past 12 months but could not afford it?'),
 ('GenHlth','General health',1,5,'1 excellent, 2 very good, 3 good, 4 fair, 5 poor.'),
 ('MentHlth','Poor mental health days',0,30,'Days of poor mental health during the past 30 days.'),
 ('PhysHlth','Poor physical health days',0,30,'Days of poor physical health during the past 30 days.'),
 ('DiffWalk','Difficulty walking',0,1,'Serious difficulty walking or climbing stairs?'),
 ('Sex','Recorded sex',0,1,'Source coding: 0 female, 1 male. Limited binary representation.'),
 ('Age','Age group',1,13,'1 = 18–24; 2 = 25–29; then five-year bands; 13 = 80+.'),
 ('Education','Education level',1,6,'1 never attended/kindergarten; 2 grades 1–8; 3 grades 9–11; 4 high school; 5 some college; 6 college graduate.'),
 ('Income','Income band',1,8,'1 under $10k; 2 $10–15k; 3 $15–20k; 4 $20–25k; 5 $25–35k; 6 $35–50k; 7 $50–75k; 8 $75k+; historical nominal dollars.'),
]
SHORT = ['HighBP','HighChol','BMI','GenHlth','Age','DiffWalk','PhysHlth','Sex']
