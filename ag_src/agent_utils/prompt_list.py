# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name : prompt_list.py 
   Description :  
   Author :       HX
   date :    2023/11/27 19:16 
-------------------------------------------------
"""

GRILQA_PROMPT = """
You need to understand a question, and plan for the process of constructing a query to solve this question.

Remeber the following RULES:
1. Never try to get a relation from a relation, for example, get_relation(measurement_unit.inverse_temperature_unit.measurement_system).

You can only choose actions from these eight actions:
1. get_relation(Freebase_mid_or_variable)
Get the one-hop relation list of a freebase mid or a variable.
2. add_fact(head,relation,tail)
Add a triple pattern: (head, relation, tail) to the query.
3. add_max(max_var)
Constrain the query by only returning the result when max_var is the biggest.
4. add_min(min_var)
Constrain the query by only returning the result when min_var is the smallest.
5. add_count(count_var,new_var)
Add a query step to count the number of elements in the variable count_var and store the result in the variable new_var. 
6. add_filter(ob1,op,ob2)
Add a filter constrain, the result need to satisfy "ob1 op ob2". ob1 and ob2 can be variables or digits, op can be >,<,>=,<=. For example, ?x,>,?y or ?x <= 0.3 is valid. 
For example, if you need ?length is greater than 10, you can use add_filter(?length,>,10) 
7. set_answer(answer_var)
For a SQL-like query, you need to determine which variable to return. Function set_answer is to determine this, it will constrain that the query only returns answer_var.
8. execute()
This is used when you think the query has been completely constructed. Calling this action will execute the query and get final answer.

Let's start with an example.

Question : which was the most recently formed cyclone that is in the same category as tropical storm rachel?
Entity : {'Tropical Storm Rachel':'m.0crbvqz'}

Thought 1 : This question aim to find a cyclone(?cyclone), this cyclone(?cyclone) has the latest formed time(?time). Besides, the category(?category) of this cyclone(?cyclone), is also the category(?category) of storm rachel(m.0crbvqz).
To construct the query, we need to first constrain ?category is the category of storm rachel(m.0crbvqz), then constrain that ?category is the category of ?cyclone, then constrain ?time is the formed time of ?cyclone, finally, filter the max ?time and only return ?cyclone.
Action 1 : get_relation(m.0crbvqz)
Observation 1 : {'forward': ['kg.object_profile.prominent_type',
  'meteorology.tropical_cyclone.category',
  'meteorology.tropical_cyclone.tropical_cyclone_season',
  'type.object.key'],
 'backward': ['meteorology.tropical_cyclone_category.tropical_cyclones',
  'meteorology.tropical_cyclone_season.tropical_cyclones',
  'type.type.instance']}

Thought 2 : Base on the relations, the meteorology.tropical_cyclone.category is the most possible relation for storm rachel(m.0crbvqz). This is a forward relation. So we constrain (m.0crbvqz, meteorology.tropical_cyclone.category, ?category) to get the ?category of storm rachel(m.0crbvqz).
Action 2 : add_fact(m.0crbvqz, meteorology.tropical_cyclone.category, ?category)
Observation 2 : "The value of variable category is ['Tropical storm (SSHS)']"

Thought 3 : By executing the current query, we get some instances of ?categoty, they seem some cyclone category. This result meets our expectation. 
Now we need to constrain that the category of ?cyclone is ?category. So let us first query for the relations of ?category.
Action 3 : get_relation(?category)
Observation 3 : {'forward': ['kg.object_profile.prominent_type',
  'meteorology.tropical_cyclone_category.max_storm_surge',
  'meteorology.tropical_cyclone_category.max_wind_speed',
  'meteorology.tropical_cyclone_category.min_storm_surge',
  'meteorology.tropical_cyclone_category.min_wind_speed',
  'meteorology.tropical_cyclone_category.tropical_cyclones',
  'type.object.key'],
 'backward': ['meteorology.tropical_cyclone.category', 'type.type.instance']}

Thought 4 : Base on the relations, the meteorology.tropical_cyclone.category is the most possible relation for ?category. This is a backward relation. So we constrain (?cyclone, meteorology.tropical_cyclone.category, ?category) to make sure the the category of ?cyclone is ?category.
Action 4 : add_fact(?cyclone, meteorology.tropical_cyclone.category, ?category)
Observation 4 : "The value of variable category is ['Tropical storm (SSHS)']. The value of variable cyclone is ['Hurricane Katrina', 'Tropical Storm Rachel', 'Tormenta tropical Arlene', 'Tempesta tropicale Lee', 'Hurricane Iselle']"

Thought 5 : By executing the current query, we get some instances of ?cyclone, they seem some cyclone. This result meets our expectation.
Now we need to query for the forming time of ?cyclone. So let us first query for the relations of ?cyclone.
Action 5 : get_relation(?cyclone)
Observation 5 : {'forward': ['kg.object_profile.prominent_type',
  'meteorology.tropical_cyclone.affected_areas',
  'meteorology.tropical_cyclone.category',
  'meteorology.tropical_cyclone.damages',
  'meteorology.tropical_cyclone.direct_fatalities',
  'meteorology.tropical_cyclone.dissipated',
  'meteorology.tropical_cyclone.formed',
  'meteorology.tropical_cyclone.highest_winds',
  'meteorology.tropical_cyclone.indirect_fatalities',
  'meteorology.tropical_cyclone.lowest_pressure',
  'meteorology.tropical_cyclone.total_fatalities',
  'meteorology.tropical_cyclone.tropical_cyclone_season',
  'type.object.key'],
 'backward': ['dataworld.gardening_hint.last_referenced_by',
  'meteorology.cyclone_affected_area.cyclones',
  'meteorology.tropical_cyclone_category.tropical_cyclones',
  'meteorology.tropical_cyclone_season.tropical_cyclones',
  'type.type.instance']}

Thought 6 : Base on the relations, the meteorology.tropical_cyclone.formed is the most possible relation for ?cyclone. This is a forward relation. So we constrain (?cyclone,  meteorology.tropical_cyclone.formed, ?time) to get the ?time when ?cyclone formed.
Action 6 : add_fact(?cyclone,  meteorology.tropical_cyclone.formed, ?time)
Observation 6 : "The value of variable ?category is ['Tropical storm (SSHS)']. The value of variable ?cyclone is ['Hurricane Katrina', 'Tormenta tropical Arlene', 'Tempesta tropicale Lee', 'Hurricane Iselle']. The value of variable ?time is ['2014-07-31-08:00', '2005-06-08-08:00', '2011-09-01-08:00', '1981-11-08:00']"

Thought 7 : By executing current query, we get some instances of ?time, they seem some time. This result meets our expectation.
Now we need to add a argmax constrain. The query only return the result with max ?time.  
Action 7 : add_max(?time)
Observation 7 : "The value of variable ?category is ['Tropical storm (SSHS)']. The value of variable ?cyclone is ['Hurricane Iselle']. The value of variable ?time is ['2014-07-31-08:00']"

Thought 8 : By executing current query, we get only one result and the ?time in this result is the biggest among all ?time shown in the last step. This result meets our expectation.
Now we need to determine which variable is the answer of this question. Based on the question, we know ?cyclone represent for the answer, so we only return this variable.
Action 8 : set_answer(?cyclone)
Observation 8 : "The value of variable ?cyclone is ['Hurricane Iselle']"

Thought 9 : This sparql seems can represent the intent of the question, execute it to get the answer.
Action 9 : execute()

"""

GRAPHQ_PROMPT = """
You need to understand a question, and plan for the process of constructing a query to solve this question.

Remeber the following RULES:
1. Always use get_relation() in the first step.
2. In Observation, if value of variable ?var contains 'UnName_Entity', it is CVT node and you can use get_relation(?var) to further constrain the query.

You can only choose actions from these eight actions:
1. get_relation(Freebase_mid_or_variable)
Get the one-hop relation list of a freebase mid or a variable.
2. add_fact(head,relation,tail)
Add a triple pattern: (head, relation, tail) to the query.
3. add_max(max_var)
Constrain the query by only returning the result when max_var is the biggest.
4. add_min(min_var)
Constrain the query by only returning the result when min_var is the smallest.
5. add_count(count_var,new_var)
Add a query step to count the number of elements in the variable count_var and store the result in the variable new_var. 
6. add_filter(ob1,op,ob2)
Add a filter constrain, the result need to satisfy "ob1 op ob2". ob1 and ob2 can be variables or digits, op can be >,<,>=,<=. For example, ?x,>,?y or ?x <= 0.3 is valid. 
For example, if you need ?length is greater than 10, you can use add_filter(?length,>,10) 
7. set_answer(answer_var)
For a SQL-like query, you need to determine which variable to return. Function set_answer is to determine this, it will constrain that the query only returns answer_var.
8. execute()
This is used when you think the query has been completely constructed. Calling this action will execute the query and get final answer.

Let's start with an example.

Question : which was the most recently formed cyclone that is in the same category as tropical storm rachel?
Entity : {'Tropical Storm Rachel':'m.0crbvqz'}

Thought 1 : This question aim to find a cyclone(?cyclone), this cyclone(?cyclone) has the latest formed time(?time). Besides, the category(?category) of this cyclone(?cyclone), is also the category(?category) of storm rachel(m.0crbvqz).
To construct the query, we need to first constrain ?category is the category of storm rachel(m.0crbvqz), then constrain that ?category is the category of ?cyclone, then constrain ?time is the formed time of ?cyclone, finally, filter the max ?time and only return ?cyclone.
Action 1 : get_relation(m.0crbvqz)
Observation 1 : 'forward': ['kg.object_profile.prominent_type', 'meteorology.tropical_cyclone.category', 'meteorology.tropical_cyclone.tropical_cyclone_season', 'type.object.key']
'backward': ['meteorology.tropical_cyclone_category.tropical_cyclones', 'meteorology.tropical_cyclone_season.tropical_cyclones', 'type.type.instance']

Thought 2 : Base on the relations, the meteorology.tropical_cyclone.category is the most possible relation. This is a forward relation. So we constrain (m.0crbvqz, meteorology.tropical_cyclone.category, ?category).
Action 2 : add_fact(m.0crbvqz, meteorology.tropical_cyclone.category, ?category)
Observation 2 : "The value of variable category is ['Tropical storm (SSHS)']"

Thought 3 : By executing the current query, we get some instances of ?categoty, they seem some cyclone category. This result meets our expectation. 
Now we need to constrain that the category of ?cyclone is ?category. So let us first query for the relations of ?category.
Action 3 : get_relation(?category)
Observation 3 : 'forward: ['kg.object_profile.prominent_type', 'meteorology.tropical_cyclone_category.max_storm_surge', 'meteorology.tropical_cyclone_category.max_wind_speed', 'meteorology.tropical_cyclone_category.min_storm_surge', 'meteorology.tropical_cyclone_category.min_wind_speed', 'meteorology.tropical_cyclone_category.tropical_cyclones', 'type.object.key']
'backward': ['meteorology.tropical_cyclone.category', 'type.type.instance']

Thought 4 : Base on the relations, the meteorology.tropical_cyclone.category is the most possible relation. This is a backward relation. So we constrain (?cyclone, meteorology.tropical_cyclone.category, ?category).
Action 4 : add_fact(?cyclone, meteorology.tropical_cyclone.category, ?category)
Observation 4 : "The value of variable category is ['Tropical storm (SSHS)']. The value of variable cyclone is ['Hurricane Katrina', 'Tropical Storm Rachel', 'Tormenta tropical Arlene', 'Tempesta tropicale Lee', 'Hurricane Iselle']"

Thought 5 : By executing the current query, we get some instances of ?cyclone, they seem some cyclone. This result meets our expectation.
Now we need to query for the forming time of ?cyclone. So let us first query for the relations of ?cyclone.
Action 5 : get_relation(?cyclone)
Observation 5 : 'forward': ['kg.object_profile.prominent_type', 'meteorology.tropical_cyclone.affected_areas', 'meteorology.tropical_cyclone.category', 'meteorology.tropical_cyclone.damages', 'meteorology.tropical_cyclone.direct_fatalities', 'meteorology.tropical_cyclone.dissipated', 'meteorology.tropical_cyclone.formed', 'meteorology.tropical_cyclone.highest_winds', 'meteorology.tropical_cyclone.indirect_fatalities', 'meteorology.tropical_cyclone.lowest_pressure', 'meteorology.tropical_cyclone.total_fatalities', 'meteorology.tropical_cyclone.tropical_cyclone_season', 'type.object.key']
'backward': ['dataworld.gardening_hint.last_referenced_by', 'meteorology.cyclone_affected_area.cyclones', 'meteorology.tropical_cyclone_category.tropical_cyclones', 'meteorology.tropical_cyclone_season.tropical_cyclones', 'type.type.instance']

Thought 6 : Base on the relations, the meteorology.tropical_cyclone.formed is the most possible relation. This is a forward relation. So we constrain (?cyclone,  meteorology.tropical_cyclone.formed, ?time).
Action 6 : add_fact(?cyclone,  meteorology.tropical_cyclone.formed, ?time)
Observation 6 : "The value of variable ?category is ['Tropical storm (SSHS)']. The value of variable ?cyclone is ['Hurricane Katrina', 'Tormenta tropical Arlene', 'Tempesta tropicale Lee', 'Hurricane Iselle']. The value of variable ?time is ['2014-07-31-08:00', '2005-06-08-08:00', '2011-09-01-08:00', '1981-11-08:00']"

Thought 7 : By executing current query, we get some instances of ?time, they seem some time. This result meets our expectation.
Now we need to add a argmax constrain. The query only return the result with max ?time.  
Action 7 : add_max(?time)
Observation 7 : "The value of variable ?category is ['Tropical storm (SSHS)']. The value of variable ?cyclone is ['Hurricane Iselle']. The value of variable ?time is ['2014-07-31-08:00']"

Thought 8 : By executing current query, we get only one result and the ?time in this result is the biggest among all ?time shown in the last step. This result meets our expectation.
Now we need to determine which variable is the answer of this question. Based on the question, we know ?cyclone represent for the answer, so we only return this variable.
Action 8 : set_answer(?cyclone)
Observation 8 : "The value of variable ?cyclone is ['Hurricane Iselle']"

Thought 9 : This sparql seems can represent the intent of the question, execute it to get the answer.
Action 9 : execute()

"""


WEBQSP_PROMPT = """
You need to understand a question, and plan for the process of constructing a query to solve this question.

You can only choose actions from these seven actions:
1. get_relation(Freebase_mid_or_variable)
Get the one-hop relation list of a freebase mid or a variable.
2. add_fact(head,relation,tail)
Add a triple pattern: (head, relation, tail) to the query.
e.g. When querying for where is azerbaijan(m.0jhd) located, constrain add_fact(m.0jhd, location.location.containedby, ?location)
3. add_max(max_var)
Constrain the query by only returning the result when max_var is the biggest.
4. add_min(min_var)
Constrain the query by only returning the result when min_var is the smallest.
5. add_time_constrain(var, relation, year)
Constrain an event to occur in certain year. relation should seem like time.event.start_date, which indidates when an event starts. year has to be in the form of as 2012.
6. set_answer(answer_var)
For a SQL-like query, you need to determine which variable to return. Function set_answer is to determine this, it will constrain that the query only returns answer_var.
7. execute()
This is used when you think the query has been completely constructed. Calling this action will execute the query and get final answer.

MUST remember the following RULES:
1. Never try to get a relation from a relation, for example, get_relation(measurement_unit.inverse_temperature_unit.measurement_system).
2. In Observation, if value of variable ?var contains 'UnName_Entity', it is CVT node and you can use get_relation(?var) to further constrain the query.
3. When the results seem to be the answer, directly use set_answer() to designate proper variable to return.
4. Add a time constraint using add_time_constrain when the question is related to specific year such as 1999.

Let's start with an example.

Question : where do most of the people live in japan
Entity : {'m.03_3d': 'Japan'}

Thought 1 : This question aims to find a place(?place) that is in Japan(m.03_3d). Besides, most of the people in japan live in this ?place. To construct the query, we need to first find all the ?place in Japan, then constrain ?people_live to be how many people live in each place, filter the max ?people_live and only return corresponding ?place as answer.
Action 1 : get_relation(m.03_3d)
Observation 1 : [location.location.adjoin_s', 'location.location.area', 'location.location.contains', 'location.location.contains_major_portion_of', 'location.location.geolocation']

Thought 2 : Base on the above relations, to get the places contained by japan, location.location.contains is the most possible relation for japan(m.03_3d). So we constrain (m.03_3d, location.location.contains, ?place).
Action 2 : add_fact(m.03_3d, location.location.contains, ?place)
Observation 2 : "The value of variable ?place is ['Atami', 'Kyoto City University of Arts', 'Tamagashi Station', 'Kimitsu', 'Futatsui', 'Uzen-Tsubaki Station', 'Kohama Station', 'Sendai Station', 'Uwada Dam', 'Takizawa']"

Thought 3 : By executing the current query, we get some instances of ?place, they seem some place in Japan. This result meets our expectation. We need to further constrain the query to get how many people live in those places. Let's query for the relations of ?place.
Action 3 : get_relation(?place)
Observation 3 : ['organization.organization.sectors', 'periodicals.newspaper_circulation_area.newspapers', 'protected_sites.protected_site.iucn_category', 'symbols.namesake.named_after', 'topic_server.population_number', 'transportation.bridge.body_of_water_spanned', 'travel.hotel.grade']

Thought 4 : Base on the above relations, to obtain how many people live in ?place, topic_server.population_number is the most possible relation for ?place. So we constrain (?place, topic_server.population_number, ?people_live).
Action 4 : add_fact(?place, topic_server.population_number, ?people_live)
Observation 4 : "The value of variable ?place is ['Ōe', 'Yokote', 'Honkawane', 'Narusawa', 'Toba', 'Karumai', 'Kuwana', 'Kawakami', 'Nihonmatsu', 'Muroran']. The value of variable ?people_live is ['112156', '111049', '353381', '6537', '113931', '121439', '219862', '35653', '129331', '136886']"

Thought 5 : By executing the current query, we get some instances of ?people_live, they seem population. This result meets our expectation. Since the question queries the place in Japan that most people live in, thus we now need to add a argmax constrain. The query only return the result with max ?people_live.
Action 5 : add_max(?people_live)
Observation 5 : "The value of variable ?place is ['Honshu']. The value of variable ?people_live is ['103000000']"

Thought 6 : By executing current query, we get only one result and the ?people_live in this result is the largest among all ?people_live. This result meets our expectation. Now we need to determine which variable is the answer of this question. Based on the question, we know ?place represent for the answer, so we only return this variable.
Action 6 : set_answer(?place)
Observation 6 : "The value of variable ?place is ['Honshu']"

Thought 7 : This sparql seems can represent the intent of the question, execute it to get the answer.
Action 7 : execute()

"""

METAQA_3HOP_PROMPT = """
You are asked to answer this question by using some predefined functions. Here is three functions you can use.

1. relate(relation)
This function will start from current entity to another entity through the relation you specify.
You can only pass in one parameter.
The relation strictly is one of ['starred_actors', 'release_year', 'written_by', 'has_genre', 'directed_by','in_language']
2. get_relation()  
Get the relation of current entity. Do not pass in any parameter, the program can automatically determinant. 
3. execute() 
This is used when you think the query has been completely constructed. Calling this action will execute the query and get final answer.

Here is some RULES you should follow strictly:
1. You should exactly hop 3 times(relate->relate->relate) which means you need choose relations 3 times and use relate(relation) 3 times. Because the question is a 3-hop question, similary to the following example.
2. The first and second relation you choosed must be the same. 
Because to solve the question you need first find the movie with the same director/writer/actor as the movie mentioned in the question. 
This is implement by first find the director/writer/actor of the movie then use the same relation to find the movie with these director/writer/actor
 

Question : who are the directors of the movies written by the writer of [The Green Mile]	
Entity: The Green Mile
Relation for the entity: ['directed_by', 'written_by', 'starred_actors']
  
Thought 1 : 
This question aims to find the directors of a MOVIE, the movies is writen by an PERSON, and the PERSON is the writer of The Green Mile.
So we need to first choose a relation represent for writer to get the PERSON, then use the same relation (writer) to get the MOVIE, finally choose a relation represent for director to get the answer.
I have already got the all relation of The Green Mile, that is ['directed_by', 'written_by', 'starred_actors'].
Based on these relation, the relation 'written_by' is most likely to represent for "writer". 
So let's going through this relationship (writtern_by) by using relate(writtern_by) and see what it linked to
Action 1 : relate(writtern_by)
Observation 1 : ['Frank Darabont', 'Stephen King']

Thought 2 : Base on the observation, we got an un-empty list, these choice seems correct. Now we need further see what relation these entity has. 
Action 2 : get_relation()
Observation 2 : ['directed_by', 'written_by']

Thought 3 : Based on the observation, we need to find a relation represent for "written by", the "written_by" is the most likely one. 
So let's keep going through this relationship (writtern_by) by using relate(writtern_by) and see what it linked to
Action 3 : relate(writtern_by)
Observation 3 : ['Maximum Overdrive',
 'The Shawshank Redemption',
 'Secret Window',
 'The Blob',
 'The Mangler',
 'Firestarter',
 'Children of the Corn',
 'The Green Mile',
 'The Night Flier',
 ]


Thought 4 : Base on the observation, we got an un-empty list, these choice seems correct. Now we need further see what relation these entity has. 
Action 4 : get_relation()
Observation 4 : ['release_year',
 'has_imdb_votes',
 'directed_by',
 'written_by',
 'starred_actors',
 'has_tags',
 'has_genre']

Thought 5 : In this step we need to find a relation represent for "the directors of". Based on the observation, the "directed_by" is the most likely one.
So let's  going through this relationship (directed_by) by using relate(directed_by) and see what it linked to
Action 5 : relate(directed_by)
Observation 5 : ['John Carpenter',
 'David Cronenberg',
 'Taylor Hackford',
 'Ralph S. Singleton', 
 'Mary Lambert',
 'Tobe Hooper',
 'Brett Leonard',
 'David Koepp',
 'William Wyler',
 'Rob Reiner']

Thought 6 : Now we have construct the whole query and can represent the intent of the question, execute it to get the answer.
Action 6 : execute()

"""

# Here is some RULES you need to strictly follow:
# 1. Do not add_condition(column_name, operator, value) on the same column twice. If you restrict col1 = A and then restrict col1 = B, you will get empty result.
# 2. Before you use add_condition(column_name, operator, value), you should first see what is in this column by use get_column(column_name).  Otherwise the value you specified may not exist in this table since you donnot know what is in this column.
#
WIKISQL_PROMPT = """
This is a text2sql task, you need to construct a sql query to get the answer of the question from a table.
I will give you a question and the column name of the table(Table Header).
You need to understand the question and use some predefined functions to construct the SQL query step-by-step.

Here is the the functions you can invoke:
1. get_column(column_name)
Get the contents of the column "column_name" in the table. 
Since the keywords in the question may be informal and cannot be found directly in the table exactly as they appear.
You can use this function to see what items are in this colum and determine which item is the associated keyword in question mostly refer to. 

2. add_condition(column_name, operator, value)
Add a restriction that the sql can only return records that satisfy this condition: "column_name  operator  value".
The column_name needs to be the name of one of the columns in the table. I will provide the name of all columns in the table in the "Table Header", just after the "Question"
The operator can only be one of: =, <, >. 
The value is the value of a cell in the table. As we mentioned, get_column(column_name) can gives you what is in this column and you can only choose this parameter from it.  
 
3. set_answer(column_name, aggregation_type)
This is to specify the answer of this sql. What column you want to return (column_name) and what aggregation function (aggregation_type) you need to perform on it. 
The column_name is the column the sql should return.
The aggregation_type indicate what aggregation need to perform on the column_name. You can only choose: None, MAX, MIN, COUNT, SUM, AVG. 
If the question just need to return the content of the column corresponding to column_name, and do not need to do anything else with it, just set the aggregation_type to None.
If the question need to return the number of all item in the column corresponding to column_nam, set the aggregation_type to COUNT.
If the question need return the maximum or minimum item among all item the column corresponding to column_nam, set the aggregation_type to MAX or MIN. 
If the question need to return the summation or average of all item in column_name, set the aggregation_type to SUM or AVG.

4. execute()
When whole query is constructed, use execute() to get the final answer.

Here is some RULES you need to strictly follow:
1. Do not add_condition(column_name, operator, value) on the same column twice. If you restrict col1 = A and then restrict col1 = B, you will get empty result.
2. Before you use add_condition(column_name, operator, value), you should first see what is in this column by use get_column(column_name).  Otherwise the value you specified may not exist in this table since you donnot know what is in this column.


Here is an example, each time i will provide a question and the table
Question : What is the total number of lyricist where the lyrics theme is romance and the song lasts 3:50?
Table Header : ['Date', 'Album name', 'Track', 'Track title', 'Lyricist', 'Music genre/style', 'Major instrument(s)', 'Lyrics theme/style', 'Duration']
  
Thought 1 : This question ask for the total number of lyricist. 
The data need to satisfied two condition: the lyrics theme should be romance AND the song should lasts 3:50.
To solve this question, we need to first require the [lyrics theme] equal to [romance].
second we require the [last] equal to [3:50].
Finally we require the answer is the COUNT of [lyricist].
So let us start from require the [lyrics theme] equal to [romance].
Based on the Table Header, i think the most possible column name related to [lytics theme] is [Lyrics theme/style].
We now need to get the content in the [Lyrics theme/style] column in this table to determine what is the [romance] most probabely refer to.
So We use get_column(Lyrics theme/style).
Action 1 : get_column(Lyrics theme/style)
Observation 1 : ['Academic failure',
 'Beauty, Attraction',
 'Bible',
 'Fantasy, Warcraft',
 'Friendship',
 'Happiness, Memories, Family, Heart',
 'How to Get Famous',
 'Humor',
 'Revenge, combat',
 'Romance',
 'Romance, Regret',
 'Romance, Sigh',
 'War, friendship, death']


Thought 2 : Base on the observation, i think the [romance] mostly refer to [Romance]. 
So now we can constrain [Lyrics theme/style] = [Romance] by using add_condition(Lyrics theme/style, =, Romance).
Action 2 : add_condition(Lyrics theme/style, =, Romance)
Observation 2 : [['2000-11-07',
  'Jay 周杰倫同名專輯',
  '1-01',
  'Adorable Woman 可愛女人 (kě ài nǚ rén)',
  'Vivian Hsu 徐若瑄',
  'R&B',
  'Guitar ( acoustic )',
  'Romance',
  '3:59'],
 ['2000-11-07',
  'Jay 周杰倫同名專輯',
  '1-02',
  'Perfection 完美主義 (wán měi zhǔ yì)',
  'Vincent Fang 方文山',
  'R&B',
  'Piano',
  'Romance',
  '4:04'],
]

Thought 3 : Now we need to consider the second condition: the [last] equal to [3:50].
Based on the Table Header, i think the most possible column name related to [last] is [Duration].
We now need to get the content in the [Duration] column in this table to determine what is the [3:50] most probabely refer to.
So We use get_column(Duration).
Action 3 : get_column(Duration)
Observation 3 : {'2.42',
 '2:35',
 '2:47',
 '3:13',
 '3:14',
 '3:46',
 '3:49',
 '3:50',
 '3:51',
 '3:52',
 '4:00',
 '4:01',
 '5:30',
 '5:39'}

Thought 4 : Base on the observation, i think the [3:50] mostly refer to [3:50]. 
So now we can constrain [Duration] = [3:50] by using add_condition(Duration, =, 3:50).
Action 4 : add_condition(Duration, =, 3:50)
Observation 4 : [['2001-09-01',
  'Fantasy 范特西',
  '2-01',
  'Love Before A.D. 愛在西元前',
  'Vincent Fang 方文山',
  'R&B',
  'Guitar ( acoustic )',
  'Romance',
  '3:50']]
  
Thought 5 : Now both two condition have been considered, we only need to specify the answer is the COUNT of [lyricist]. 
Based on the Table Header, i think the most possible column name related to [lyricist] is [lyricist]. 
So we use set_answer(lyricist, COUNT)
Action 5 : set_answer(lyricist, COUNT)
Observation 5 : [1]


Thought 6 : The whole query has be construct, execute and get the final answer. 
Action 6 : execute()

"""
