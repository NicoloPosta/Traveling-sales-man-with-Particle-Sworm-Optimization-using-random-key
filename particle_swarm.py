import random
import tsplib95
import customtkinter as ctk
from tkinter import filedialog
from sklearn.preprocessing import minmax_scale


def get_optimal_path(file_path):
    solution_file_path = file_path[:-3] + "opt.tour"
    solution = tsplib95.load(solution_file_path).tours[0]
    if tsplib95.load(file_path).edge_weight_format == "LOWER_DIAG_ROW":
        solution = [x - 1 for x in solution]
    return solution

def read_tsp_instance(file_path):
    problem = tsplib95.load(file_path)
    cities = list(problem.get_nodes())
    distances = {(i, j): problem.get_weight(i, j) for i in cities for j in cities if i != j}
    return cities, distances

def initialize_particles(num_particles, num_cities):
    particles = [{'position': [random.random() for _ in range(num_cities)],
                  'velocity': [random.random() for _ in range(num_cities)],
                  'fitness': float('inf'), 
                  'best_position': None,
                  'best_fitness': float('inf')} for _ in range(num_particles)]
    return particles

def random_key_to_tsp_solution(random_key, cities):
    sorted_cities = [city for _, city in sorted(zip(random_key, cities))]
    return sorted_cities

def tsp_fitness(solution, distances):
    total_distance = sum(distances[solution[i - 1], solution[i]] for i in range(len(solution)))
    # Aggiungo l'arco dall'ultima alla prima città quando i = 0
    return total_distance

def update_particles(particles, best_particle, w, c1, c2):
    for particle in particles:
        for i in range(len(particle['position'])):
            r1, r2 = random.random(), random.random()
            particle['velocity'][i] = w * particle['velocity'][i] + c1 * r1 * (best_particle['position'][i] - particle['position'][i]) + c2 * r2 * (particle['best_position'][i] - particle['position'][i])
            particle['position'][i] += particle['velocity'][i]

        particle['position'] = minmax_scale(particle['position'], feature_range=(0,1))

def pso_tsp(file_path, num_particles, max_iterations, w, c1, c2, variable_w, result_label=None, window=None, progress_bar=None):
    progress_bar.set(0)
    window.update()
    print(f"Now running with: w = {w}, c1 = {c1}, c2 = {c2}, num_particles = {num_particles}, max_iterations = {max_iterations}, variable w {variable_w}")
    if variable_w:
        diminishing_rate = ((w) / max_iterations)
    cities, distances = read_tsp_instance(file_path)
    num_cities = len(cities)

    particles = initialize_particles(num_particles, num_cities)
    global_best_particle = {'position': None, 'fitness': float('inf')}
    last_update_iteration = 0
    
    for iteration in range(max_iterations):
        if iteration == max_iterations-1:
            progress_bar.set(100)
            window.update()
        elif iteration % (max_iterations / 100) == 0 and iteration != 0:
            progress_bar.step()
            window.update()
        
        for particle in particles:
            tsp_solution = random_key_to_tsp_solution(particle['position'], cities)
            particle['fitness'] = tsp_fitness(tsp_solution, distances)

            # Aggiorno la miglior posizione della particella
            if particle['fitness'] < particle['best_fitness']:
                particle['best_fitness'] = particle['fitness']
                particle['best_position'] = particle['position']

            # Aggiorno la miglior posizione globale
            if particle['fitness'] < global_best_particle['fitness']:
                last_update_iteration = iteration
                print(f"Updated best solution at iteration: {iteration} with fitness of {particle['fitness']}")
                result_label.configure(text=f"Running...\nUpdated best solution at iteration: {iteration} with fitness of {particle['fitness']}")
                window.update()
                global_best_particle['fitness'] = particle['fitness']
                global_best_particle['position'] = particle['position']

        # Calcolo il valore attuale di w (diminuendo linearmente da initial_w a 0)
        if variable_w:
            w = w - diminishing_rate
        
        # Aggiorno le posizioni delle particelle
        update_particles(particles, global_best_particle, w, c1, c2)

    # La migliore soluzione è rappresentata dalla miglior particella
    best_solution = random_key_to_tsp_solution(global_best_particle['position'], cities)
    best_fitness = global_best_particle['fitness']
    optimal_solution = get_optimal_path(file_path)
    optimal_cost = tsp_fitness(optimal_solution, distances)
    error = ((best_fitness - optimal_cost) / optimal_cost) * 100

    return best_solution, best_fitness, optimal_solution, optimal_cost, error, last_update_iteration

def create_gui():
    window = ctk.CTk()
    window.resizable(False,False)
    window.title("PSO TSP Solver")
    print_solution_var = ctk.BooleanVar(value=False)
    print_optimum_var = ctk.BooleanVar(value=False)
    variable_w = ctk.BooleanVar(value=False)
    confinement = ctk.BooleanVar(value=False)

    # Funzione chiamata quando si preme il pulsante Run PSO
    def run_pso():
        file_path = file_path_var.get()
        if file_path == '':
            result_label.configure(text="Select a file before running the algorithm")
            return
        num_particles = int(num_particles_entry.get())
        if num_particles <= 0:
            result_label.configure(text="Select a value greather than 0 for the number of particles")
            return
        max_iterations = int(max_iterations_entry.get())
        if max_iterations <= 0:
            result_label.configure(text="Select a value greather than 0 for the number of iterations")
            return
        w = float(w_entry.get())
        if w < 0 or w > 1.0:
            result_label.configure(text="Select a value between 0 and 1 for the variable w")
            return
        c1 = float(c1_entry.get())
        if c1 < 0.0 or c1 > 2.0:
            result_label.configure(text="Select a value between 0 and 2 for the variable c1")
            return
        c2 = float(c2_entry.get())
        if c2 < 0.0 or c2 > 2.0:
            result_label.configure(text="Select a value between 0 and 2 for the variable c2")
            return

        result_label.configure(text="Running...")
        window.update()
        best_solution, best_fitness, optimal_solution, optimal_cost, error, last_update_iteration = pso_tsp(file_path, num_particles, max_iterations, w, c1, c2, variable_w.get(), result_label, window, progress_bar)

        result_text = f"TSP file name: {file_path.split('/')[-1]}\nLast update of the fitness at iteration: {last_update_iteration}\nBest TSP fitness found: {best_fitness}\nOptimum TSP fitness: {optimal_cost}\nError from optimum: {error:.2f}%"

        if len(best_solution) > 60:
            num_elem = 40
        else:
            num_elem = 15
        if print_solution_var.get():
            solution_string = "\n".join([f"{best_solution[i:i+num_elem]}" for i in range(0, len(best_solution), num_elem)])
            result_text += f"\nBest TSP solution: {solution_string}"
            
        if print_optimum_var.get():
            optimal_solution_string = "\n".join([f"{optimal_solution[i:i+num_elem]}" for i in range(0, len(optimal_solution), num_elem)])
            result_text += f"\nOptimum TSP solution: {optimal_solution_string}"
        
        result_label.configure(text=result_text)  
    
    # Interfaccia utente
    file_path_var = ctk.StringVar()
    file_path_label = ctk.CTkLabel(window, text="TSP File:")
    file_path_entry = ctk.CTkEntry(window, textvariable=file_path_var, width=125)
    file_path_entry.configure(state="disabled")
    file_path_button = ctk.CTkButton(window, text="Browse", command=lambda: browse_file(file_path_var))

    num_particles_label = ctk.CTkLabel(window, text="Number of particles:")
    num_particles_entry = ctk.CTkEntry(window, textvariable=ctk.StringVar(value=num_particles), width=60)

    max_iterations_label = ctk.CTkLabel(window, text="Max iterations:")
    max_iterations_entry = ctk.CTkEntry(window, textvariable=ctk.StringVar(value=max_iterations), width=60)

    w_label = ctk.CTkLabel(window, text="(Coefficient of inertia) w:")
    w_entry = ctk.CTkEntry(window, textvariable=ctk.StringVar(value=w), width=40)
    
    variable_w_checkbox = ctk.CTkCheckBox(window, text="Variable w", variable=variable_w)

    c1_label = ctk.CTkLabel(window, text="(Self best) c1:")
    c1_entry = ctk.CTkEntry(window, textvariable=ctk.StringVar(value=c1), width=40)

    c2_label = ctk.CTkLabel(window, text="(Global best) c2:")
    c2_entry = ctk.CTkEntry(window, textvariable=ctk.StringVar(value=c2), width=40)
    
    print_solution_checkbox = ctk.CTkCheckBox(window, text="Print solution", variable=print_solution_var)

    print_optimum_checkbox = ctk.CTkCheckBox(window, text="Print Optimum solution", variable=print_optimum_var)

    run_button = ctk.CTkButton(window, text="Run PSO", command=run_pso)
    
    progress_bar = ctk.CTkProgressBar(window, orientation="orizontal", width=300, determinate_speed=0.5)
    progress_bar.set(0)

    result_label = ctk.CTkLabel(window, text="Results will be displayed here.")

    # Layout dell'interfaccia utente
    file_path_label.grid(row=0, column=0, padx=10, pady=5, sticky=ctk.E)
    file_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky=ctk.W)
    file_path_button.grid(row=0, column=2, padx=5, pady=5)

    num_particles_label.grid(row=1, column=0, padx=10, pady=5, sticky=ctk.E)
    num_particles_entry.grid(row=1, column=1, padx=5, pady=5, sticky=ctk.W)

    max_iterations_label.grid(row=2, column=0, padx=10, pady=5, sticky=ctk.E)
    max_iterations_entry.grid(row=2, column=1, padx=5, pady=5, sticky=ctk.W)
    
    w_label.grid(row=4, column=0, padx=10, pady=5, sticky=ctk.E)
    w_entry.grid(row=4, column=1, padx=5, pady=5, sticky=ctk.W)
    
    variable_w_checkbox.grid(row=4, column=2, padx=5, pady=5, sticky=ctk.W)

    c1_label.grid(row=5, column=0, padx=10, pady=5, sticky=ctk.E)
    c1_entry.grid(row=5, column=1, padx=5, pady=5, sticky=ctk.W)

    c2_label.grid(row=6, column=0, padx=10, pady=5, sticky=ctk.E)
    c2_entry.grid(row=6, column=1, padx=5, pady=5, sticky=ctk.W)
    
    print_solution_checkbox.grid(row=5, column=2, columnspan=3, padx=5, pady=5, sticky=ctk.W)
    
    print_optimum_checkbox.grid(row=6, column=2, columnspan=3, padx=5, pady=5, sticky=ctk.W)

    run_button.grid(row=8, column=0, columnspan=3, pady=10)
    
    progress_bar.grid(row=9, column=0, columnspan=3, pady=10)

    result_label.grid(row=10, column=0, columnspan=3, pady=10)

    window.mainloop()

# Funzione per il pulsante di navigazione per selezionare un file TSP
def browse_file(entry_var):
    file_path = filedialog.askopenfilename(filetypes=[("TSP Files", "*.tsp")])
    entry_var.set(file_path)

if __name__ == "__main__":
    # Parametri di default
    max_iterations = 5000
    num_particles = 20
    w, c1, c2 = 0.7, 1.43, 1.43
    min_velocity, max_velocity = 100, 100

    # Avvia l'interfaccia utente
    create_gui()