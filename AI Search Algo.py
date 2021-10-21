# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 12:54:16 2020

@author: Owner
"""
# =============================================================================
# packages
# =============================================================================
import pandas as pd
import numpy as np
import math
import datetime
import random
from termcolor import colored

# =============================================================================
# ############ import data #############
# =============================================================================
counties_data = pd.read_csv(r"C:\Users\Owner\Desktop\DB\adjacency.csv")
counties_data = pd.DataFrame(counties_data,columns= ['countyname','neighborname'])
county_lon_lat = pd.read_csv(r"C:\Users\Owner\Desktop\DB\county_lon_lat.csv")
'''clean all lines that county as adjacent to itself'''
counties_data=counties_data[counties_data.countyname !=counties_data.neighborname ]
# =============================================================================
# ######### functions ##################
# =============================================================================
    
def heristic_distance(lat_from,lon_from,lat_to,lon_to):
    lat1 =pd.to_numeric(lat_from, errors='raise', downcast=None)
    lon1 =lon_from
    lat2 =pd.to_numeric(lat_to, errors='raise', downcast=None)
    lon2 =lon_to
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    
    return d/200

def heu(county1,county2):
    county1_cor = county_cordinate(county1)
    county1_lan = float(county1_cor[1])
    county1_lon = float(county1_cor[0])
    county2_cor = county_cordinate(county2)
    county2_lan = float(county2_cor[1])
    county2_lon = float(county2_cor[0])
    heu = heristic_distance(county1_lan,county1_lon,county2_lan,county2_lon)
    return heu

def find_shortname_county(countyname):
    county_short = ""
    country = countyname
    if "County" in countyname:
        split_string = country.split(" County", 1)
        substring = split_string[0]
        if "St." in substring:
            substring = substring.replace("St.", "Saint")
        county_short = substring
        return county_short
    if "city" in countyname:
        split_string = country.split(" city", 1)
        substring = split_string[0]
        county_short = substring
        return county_short
    if "City" in countyname:
        split_string = country.split(" City", 1)
        substring = split_string[0]
        county_short = substring
        return county_short
    if "Parish" in countyname:
        split_string = country.split(" Parish", 1)
        substring = split_string[0]
        county_short = substring
        return county_short
    if "Area" in countyname:
        split_string = country.split(" Area", 1)
        substring = split_string[0]
        county_short = substring
        return county_short
    if "Borough" in countyname:
        split_string = country.split(" Borough", 1)
        substring = split_string[0]
        county_short = substring
        return county_short
    if "Municipio" in countyname:
        split_string = country.split(" Municipio", 1)
        substring = split_string[0]
        county_short = substring
        return county_short
    if "District" in countyname:
        split_string = country.split(" District", 1)
        substring = split_string[0]
        county_short = substring
        return county_short
    if "Municipality" in countyname:
        split_string = country.split(" Municipality", 1)
        substring = split_string[0]
        county_short = substring
        return county_short
    if "Island" in countyname:
        split_string = country.split(" Island", 1)
        substring = split_string[0]
        county_short = substring
        return county_short
    if "Yukon-Koyukuk" in countyname:
        substring = "Yukon Koyukuk"
        county_short = substring
        return county_short
    else:
        return countyname

#find_shortname_county("Portsmouth city, VA")
def finfneighbor(countyname):
   z= counties_data[counties_data["countyname"]==countyname].index.tolist()
   neighbores = list()
   for i in z:
       neighbores.append(counties_data["neighborname"][i])
   return neighbores

def county_cordinate(county): 
    cordinate = list()
    short= find_shortname_county(county)
    index= county_lon_lat[county_lon_lat['county']==short].index.tolist()
    if index == []: #county doest found 
        cordinate.append(1000) # very big and fake coordinate so the algorithem wont go there
        cordinate.append(1000)
    else:
        cordinate.append(county_lon_lat['long'][index] )
        cordinate.append(county_lon_lat['lat'][index] )
    return cordinate


def open_county(current_county,targetname):
    open_county = list()    #return the country short wuth the huristic from the target
    neibores    = finfneighbor(current_county)
    trarget_cor = county_cordinate(targetname)
    target_lan = float(trarget_cor[1])
    target_lon = float(trarget_cor[0])
    for i in neibores:
        current_cor = county_cordinate(i)
        current_lat = float(current_cor[1])
        current_lon = float(current_cor[0])
        heu = heristic_distance(target_lan,target_lon,current_lat,current_lon)
        temp =list()
        temp.append(i)
        temp.append(heu)
        open_county.append(temp)
    return open_county

def A_Star(starting_locations, goal_locations):
    path = []             # closed nodes
    close = []
    node = starting_locations
    path.append(node)
    close.append(node)
    targetname = goal_locations
    g = 0
    
    while node != targetname:
        opened = open_county(node,targetname)
        if targetname in [county[0] for county in path]:
            path.append(targetname)
            g = g + 1
            return path
        opened = [x for x in opened if x[0] not in close]   
        if opened == [] or g > 400:
            return "No path found",path
        h = [j[1] for j in opened]
        chosen_index = h.index(min(h))
        node = opened[chosen_index][0]  # current node
        close.append(node)
        h_val = opened[chosen_index][1]
        node_and_heu =[]
        node_and_heu.append(node)
        node_and_heu.append(h_val)
        path.append(node_and_heu) 
        g = g + 1
        h = 0

    return path

def getList(dict): 
    list = [] 
    for key in dict.keys(): 
        list.append(key) 
    return list
        
 
# =============================================================================
# #################################### hill climbing ##################
# =============================================================================
def print_hill_paths(paths):
    print()
    leng = [len(j) for j in paths]
    longest_path = max(leng)
    i = 0
    while i < longest_path:
        row = "{"
        for path in paths:
            if i < len(path):
                row += path[i][0] + " ; "
            else:
                row += path[-1][0] + " ; "
        row += "}"
        print(row)
        i += 1      

def random_5_neighbores(neighbores):
    len_n= len(neighbores)
    if len_n <= 5:
        return neighbores
    chosen_neighbores =random.sample(neighbores, 5)
    return chosen_neighbores

def hill(starting_locations, goal_locations):
    path = []             # closed nodes
    node = starting_locations
    heur = heu(starting_locations, goal_locations)
    node_and_heu =[]
    node_and_heu.append(node)
    node_and_heu.append(heur)
    path.append(node_and_heu)
    targetname = goal_locations 
    opened = open_county(node,targetname)   #return all neibores with heuristics
    h = [j[1] for j in opened]
    chosen_index = h.index(min(h))          #return index of neibore with minimum heuristics
    next_node = opened[chosen_index][0]     # next node name
    next_node_heu=opened[chosen_index][1]   #next node heuristic
    while next_node_heu < heur:             #stop condition- didnt find better neibore 
        node = next_node
        heur = next_node_heu
        opened = open_county(node,targetname)
        h = [j[1] for j in opened]
        chosen_index = h.index(min(h))
        next_node = opened[chosen_index][0]  # current node
        next_node_heu=opened[chosen_index][1]
        node_and_heu =[]
        node_and_heu.append(node)
        node_and_heu.append(heur)
        path.append(node_and_heu) 
    return path
#luck = hill("Autauga County, AL","Chilton County, AL")

def hill_climbing_soulotion(start,targetname):
    potenial_path = []  # all 5 path
    neighbores = finfneighbor(start)
    chosen_neighbores = random_5_neighbores(neighbores)
    for neighbor in chosen_neighbores:
        path = []
        node_and_heu =[]
        node_and_heu.append(start)
        heur = heu(start,targetname)
        node_and_heu.append(heur)
        path.append(node_and_heu)
        hil = hill(neighbor,targetname)
        for i in hil:
           path.append(i) 
        potenial_path.append(path)
    h = [j[len(j)-1][1] for j in potenial_path]
    chosen_index = h.index(min(h))
    choosen_path = potenial_path[chosen_index]
    return choosen_path

#my_s = hill_climbing_soulotion("Autauga County, AL","Chilton County, AL")

# =============================================================================
# #################################### Simulated Annealing ##################
# =============================================================================
def simulated_annealing(start,targetname):
    """Peforms simulated annealing to find a solution"""
    path = []
    initial_temp = 100
    final_temp = 0.3
    alpha = 0.94
    ite = 0
    deatail_output = []
    
    current_temp = initial_temp

    # Start by initializing the current state with the initial state
    current_county = []
    current_county.append(start)
    current_county.append(heu(start,targetname))
    path.append(current_county)

    while current_temp > final_temp:
        
        ite = ite+1
        neighbor = random.choice(open_county(current_county[0],targetname))
        if neighbor[0] == targetname:
             path.append(neighbor)
             break
        # Check if neighbor is best so far
        cost_diff = round(current_county[1] - neighbor[1])
        
        # if the new solution is better, accept it
        if cost_diff >= 0:
            path.append(neighbor)
            current_county = neighbor
            
            
        # if the new solution is not better, accept it with a probability of e^(-cost/temp)
        else:
            r = random.uniform(0, 1)
            m = math.exp(cost_diff / current_temp)
            action  = "The difference between the heuristics is:  " + str(round(cost_diff)) +", " + " i consider going to " + neighbor[0] + " in probability of :" + str(m)
            deatail_output.append(action)
            if r < m:
                path.append(neighbor)
                current_county = neighbor
                

        # decrement the temperature
        
        current_temp = current_temp*alpha
      
    return path, deatail_output #path,ite
    
    
#sim, details = simulated_annealing("Autauga County, AL","Hinds County, MS")
#sim,ite = simulated_annealing("Chesapeake city, VA","Hinds County, MS")
# =============================================================================
# #################################### local beam search ##################
# =============================================================================
def takeSecond(elem):
    return elem[1]

def Remove(duplicate): 
    final_list = []
    for c in duplicate:
        temp_counties = [x[0] for x in final_list]
        if c[0] not in temp_counties:
            final_list.append(c)
    return final_list


def local_beam_search(start,target):
    bag = []
    opened = open_county(start,target)
    opened.sort(key=takeSecond)
    k1,k2, k3, t1 , t2 , t3  =[[] for _ in range(6)]
    K = [k1,k2,k3]
    for k in K:
        node = start
        heur = heu(start, target)
        node_and_heu =[]
        node_and_heu.append(node)
        node_and_heu.append(heur) 
        k.append(node_and_heu)
    k1.append(opened[0])
    k2.append(opened[1])
    k3.append(opened[2])
    all_options = []
    temp_k = [t1,t2,t3]
    close = []
    best_path_index = 0
    last_counties = [j[len(j)-1][0] for j in K] 
    while target not in last_counties:
      
        for k in K:
            neibor = open_county(k[len(k)-1][0],target)
            for n in neibor:    #In order to reconstruct where the districts came from we must mark them
                if k is k1:
                    n.append(1)
                    if n[0] == target:
                        k.append(n)
                        return k,bag
                if k is k2:
                    n.append(2)
                    if n[0] == target:
                        k.append(n)
                        return k,bag
                if k is k3:
                    n.append(3)
                    if n[0] == target:
                        k.append(n)
                        return k,bag
                all_options.append(n)
        c = []        
        c = Remove(all_options)   #Make sure there are no duplicates
        all_options = [x for x in c if x[0] not in close]
        option_names = [j[0] for j in all_options]
        bag.append(option_names)    # for detail output
        all_options.sort(key=takeSecond)
        if len(all_options)<3:
            break
        l = len(all_options)-1
        for i in range(2,l):
                    all_options.pop(3)  #stay in all_options with the best counties -set
        [close.append(j[0]) for j in all_options ]
        for i in range(3):
            if all_options[i][2] == 1:
                temp_k[i] = k1.copy()
                temp_k[i].append(all_options[i])
            if all_options[i][2] == 2:
                temp_k[i] = k2.copy()
                temp_k[i].append(all_options[i])
            if all_options[i][2] == 3:
                temp_k[i] = k3.copy()
                temp_k[i].append(all_options[i])
        k1 = temp_k[0].copy()
        k2 = temp_k[1].copy()
        k3 = temp_k[2].copy()
        K = [k1,k2,k3]
        temp_k = [t1,t2,t3]
    last_counties = [j[len(j)-1][0] for j in K]
    best_path_index = last_counties.index(min(last_counties))
    return K[best_path_index], bag
   

#g,b = local_beam_search("Hancock County, WV","Platte County, MO")      
    
    
# =============================================================================
# ################################### genetic algorithem ################
# =============================================================================
def print_hill_genetic(paths):
    print()
    leng = [len(j) for j in paths]
    longest_path = max(leng)
    i = 0
    while i < longest_path:
        row = "{"
        for path in paths:
            if i < len(path):
                row += path[i] + " ; "
            else:
                row += path[-1] + " ; "
        row += "}"
        print(row)
        i += 1      
        
def create_population(start,target):
    s_population = []
    f_population = []
    
    for i in range(5):
        
        s = []
        s.append(start)
        ran_s = random.sample(list(counties_data.index),10)
        for r in ran_s:
            s.append(counties_data["countyname"][r])
        s_population.append(s)
        f = []
        ran_f = random.sample(list(counties_data.index),10)
        for r in ran_f:
             f.append(counties_data["countyname"][r])
        f.append(target)
        f_population.append(f)
        population = s_population + f_population
    return population



def genetic_score(path,target):
    score = 0
    for i in range(1,len(path)-1):
        opend= open_county(path[i-1],target)
        neibors = [j[0] for j in opend]
        if path[i] in neibors:
            score = score + 1
    return score



def find_x(population,target):
    Cumulative = 0
    index      = 0
    df = pd.DataFrame(columns=['path', 'score','probability','Cumulative probability'])
    for i in range(len(population)):
        df.at[i,'path'] = population[i]      
        genetic_scor = genetic_score(population[i],target)
        if genetic_scor == 0 :
            genetic_scor = random.uniform(0, 0.1)
        df.at[i,'score'] = genetic_scor 
    sump = df['score'].sum() 
    if sump == 0:
        sump = 1
    for i in range(len(population)):
        sco = df['score'][i]
        pro = sco / sump
        df.at[i,'probability'] = pro
        Cumulative = Cumulative + df['probability'][i]
        df.at[i,'Cumulative probability'] = Cumulative
    c = random.random()
    for i in range(len(population)):
        if df["Cumulative probability"][i]>c:
            index = i
            break
    choosen_path = df["path"][index]
    return choosen_path



def REPRODUCE(x, y):
    new_child = []
    n = len(x)
    c = random.randrange(1,n)
    new_childx = x[:c]
    new_childy = y[c:n]
    new_child = new_childx + new_childy
    return new_child

def MUTATE(child):
    n = len(child)
    c = random.randrange(1,n)
    e = counties_data.sample()
    new_county = e.iat[0,0]
    child[c] = new_county
    return child

def GENETIC_ALGORITHM(start,target):
    population_num = 0
    output_detail = {}
    endTime = datetime.datetime.now() + datetime.timedelta(minutes=0.1)
    population = create_population(start,target)
    p = [p for p in population ]
    output_detail.update({population_num:p})
    while True:
        if datetime.datetime.now() >= endTime:
            break
    new_population = []
    for i in range(len(population)):  #SAME NUMBER OF CHILDREN
                x = find_x(population,target)
                y = find_x(population,target)
                child = REPRODUCE(x, y)
                c = random.random()
                if (c < 0.1):
                  child = MUTATE(child) 
                new_population.append(child)
    population_num = population_num + 1
    population = new_population.copy()
    p = [p for p in population ]
    output_detail.update({population_num:p})
    sco = [genetic_score(k,target) for k in population ]
    best_path_index = sco.index(max(sco))
    
    return population[best_path_index],output_detail

#gen,output_det = GENETIC_ALGORITHM("Fairfield County, CT","Rensselaer County, NY")
# =============================================================================
# #################### find path function  ###################
# =============================================================================
def find_path(starting_locations, goal_locations,search_method,detail_output):
    if search_method ==  1:
        allpath = list()
        start_county = []
        for i in range(len(starting_locations)):
            s_county = starting_locations[i]
            start_county.append(s_county)
            f_county = goal_locations[i]
            ans = A_Star(s_county, f_county)
            if ans == "No path found":
                return "No path found"
            allpath.append(ans)
        print("{"+" %s" % start_county +"}")
        h = [len(j) for j in allpath]
        max_path = max(h)
        for i in range(1,max_path,1):
            new_dict = {}
            for path in allpath:
                if len(path) <= i :
                    a =path[len(path)-1][0]
                    b =path[len(path)-1][1]
                    new_dict[a] = b
                    continue
                else:
                    a=path[i][0]
                    b=path[i][1]
                    new_dict[a]= b
            if detail_output == True:
                print(new_dict)
            else:
                only_names = getList(new_dict)
                print ("{" + "%s" %  only_names + "}")
        return new_dict
    if search_method ==  2:
        ans = []
        for i in range(len(starting_locations)):
            s_county = starting_locations[i]
            f_county = goal_locations[i]
            hill = hill_climbing_soulotion(s_county,f_county)
            ans.append(hill)
        print_hill_paths(ans)
    if search_method ==  3:
        if detail_output == False:
            ans = []
            for i in range(len(starting_locations)):
                s_county = starting_locations[i]
                f_county = goal_locations[i]
                sim,d = simulated_annealing(s_county,f_county)
                ans.append(sim)
            print_hill_paths(ans)
        else:
            ans = []
            for i in range(len(starting_locations)):
                s_county = starting_locations[i]
                f_county = goal_locations[i]
                sim,details = simulated_annealing(s_county,f_county)
                ans.append(details)
            for i in range(len(ans)):
                text1 = colored(f"path number - {i} from {starting_locations[i]} to {goal_locations[i]}", 'blue', attrs=['underline','bold'])
                print(text1)
                print()
                for j in ans[i]:
                     print(j)
                     print()
    if search_method ==  4:
        if detail_output == False:
            ans = []
            for i in range(len(starting_locations)):
                s_county = starting_locations[i]
                f_county = goal_locations[i]
                beam, bag = local_beam_search(s_county,f_county)
                ans.append(beam)
            print_hill_paths(ans)  
        else:
            ans = []
            for i in range(len(starting_locations)):
                s_county = starting_locations[i]
                f_county = goal_locations[i]
                beam, bag = local_beam_search(s_county,f_county)
                ans.append(bag)
            for i in range(len(ans)):
                text1 = colored(f"path number - {i} from {starting_locations[i]} to {goal_locations[i]}", 'blue', attrs=['underline','bold'])
                print(text1)
                print()
                g = 0
                for b in ans[i]:
                    text1 = colored(f"the options in Action number {g} is:", 'red', attrs=['underline','bold'])
                    print(text1)
                    print()
                    list_of_options = [j for j in b]  
                    print(" %s " % list_of_options ) 
                    print()
                    g = g + 1
            return ans       
    if search_method ==  5:
        if detail_output == False:
            ans = []
            for i in range(len(starting_locations)):
                s_county = starting_locations[i]
                f_county = goal_locations[i]
                gen,detail_out = GENETIC_ALGORITHM(s_county,f_county)
                ans.append(gen)
            print_hill_genetic(ans)   
        else:
            ans = []
            for i in range(len(starting_locations)):
                s_county = starting_locations[i]
                f_county = goal_locations[i]
                gen,detail_out = GENETIC_ALGORITHM(s_county,f_county)
                ans.append(detail_out)
            for population in ans:
                for key, values in population.items():
                    i = 0
                    print(colored("population for iteration - " + str(key),'blue', attrs=['underline','bold']))
                    print()
                    for v in values:
                        print(colored( "path number - " + str(i),'red'))
                        print(v)
                        print()
                        i = i + 1
        
# =============================================================================
# ################################# import starting_locations and goal_locations #####################
# =============================================================================
starting_locations = ["Dallas County, AL" , "Baldwin County, AL" , "Baldwin County, AL" ]
goal_locations = ["Chilton County, AL" , "Cullman County, AL" , "Rensselaer County, NY"]  

# =============================================================================
# ############### run for answer ###########################    
# =============================================================================
final = find_path(starting_locations, goal_locations,3,False)

