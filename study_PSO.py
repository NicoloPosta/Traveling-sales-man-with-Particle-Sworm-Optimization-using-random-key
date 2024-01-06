from particle_swarm import *
import pandas as pd
import pathlib
import concurrent.futures
import sys
import threading
import psutil
import os


def pso_tsp_thread(file_path, num_particles, max_iterations, w, c1, c2, variable_w, confinement):
    if variable_w:
        diminishing_rate = ((w) / max_iterations)
    cities, distances = read_tsp_instance(file_path)
    num_cities = len(cities)

    particles = initialize_particles(num_particles, num_cities)
    global_best_particle = {'position': None, 'fitness': float('inf')}
    last_update_iteration = 0
    
    for iteration in range(max_iterations):     
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

def terminate_children():
    # Elimino tutti i processi figli generati dal programma
    current_process = psutil.Process(os.getpid())
    children = current_process.children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass

def run_tsp_parallel(problem_path, num_particles, max_iterations, w, c1, c2, variable_w):
    _, best_fitness, _, optimal_cost, error, last_update_iteration = pso_tsp_thread(problem_path, num_particles, max_iterations, w, c1, c2, variable_w)
    data_row = pd.DataFrame({"best_fitness": [best_fitness], "optimal_cost": [optimal_cost], "error": [error], "last_update_iteration": [last_update_iteration]})
    return data_row

def study_gui():
    window = ctk.CTk()
    window.resizable(False,False)
    window.title("PSO study TSP Solver")
    variable_w = ctk.BooleanVar(value=False)
    
    def run_pso_study():
        # Creo un thread per eseguire la funzione in background e lasciare la finestra utilizzabile
        thread = threading.Thread(target=pso_study)
        thread.start()
    
    def chiudi_finestra():
        print("Exit requested. Wait for the termination of the processes.")
        terminate_children()
        window.destroy()
        sys.exit(0)
    
    window.protocol("WM_DELETE_WINDOW", chiudi_finestra)

    def pso_study():
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
        num_iterations = int(num_iterations_entry.get())
        if num_iterations <= 0:
            result_label.configure(text="Select a value greather than 0 for the number of iterations")
            return
        
        variable_w_value = variable_w.get()
        
        # Disabilito il pulsante durante l'esecuzione della funzione
        run_button.configure(state="disabled")
        
        result_label.configure(text="Running...")
        window.update()

        problems_paths = list(pathlib.Path('Problems').glob('*.tsp'))

        results = {problem.name: {"solutions": pd.DataFrame(columns=["best_fitness", "optimal_cost", "error", "last_update_iteration"])} for problem in problems_paths}

        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            num_iterations_ref = num_iterations
            if num_iterations < 4:
                num_iterations_ref = 4
            speed = float(100) / float(len(problems_paths) * int(num_iterations_ref / 4))
            progress_bar.configure(determinate_speed=speed/2)
            progress_bar.set(0)
            for problem_path in problems_paths:
                futures = []
                problem_name = str(problem_path.name)
                print(f"Study of problem {problem_name} started...")
                results[problem_name]["solutions"] = results[problem_name]["solutions"].dropna(axis=1, how='all')

                for i in range(0, num_iterations, 4):
                    # Seleziona 4 iterazioni per l'iterazione corrente
                    current_iterations = range(i, min(i + 4, num_iterations))
                    try:
                        for j in current_iterations:
                            # Faccio partire i 4 processi figli
                            print(f"Created process n.{j} for {problem_name}")
                            future = executor.submit(run_tsp_parallel, str(problem_path.absolute()), num_particles, max_iterations, w, c1, c2, variable_w_value)
                            futures.append(future)

                        # Attendo il termine dei processi correnti
                        concurrent.futures.wait(futures, timeout=None, return_when=concurrent.futures.FIRST_EXCEPTION)

                        for future in futures:
                            # Concateno i DataFrames
                            results[problem_name]["solutions"] = pd.concat([results[problem_name]["solutions"], future.result()])
                    
                    except KeyboardInterrupt:
                        print("Exit requested. Wait for the termination of the processes.")
                        terminate_children()
                        sys.exit(0)
                    progress_bar.step()
            progress_bar.set(100)

        result_string = f"Total trials: {num_iterations*len(problems_paths)} each problem was solved {num_iterations} times.\n"

        for result in results:
            mean_fitness = results[result]["solutions"]["best_fitness"].mean()
            mean_error = results[result]["solutions"]["error"].mean()
            optimal_cost = results[result]["solutions"]["optimal_cost"].iloc[0]
            mean_last_update_iteration = results[result]["solutions"]["last_update_iteration"].mean()
            result_string += f"Problem: {result} Num iterations per solve: {max_iterations}\n   Mean fitness: {mean_fitness:.1f} Optimal cost: {optimal_cost} Mean error: {mean_error:.1f} Mean last update iteration: {mean_last_update_iteration:.1f}   \n"
        
        print(result_string)
        result_label.configure(text=result_string)
        window.update()
        run_button.configure(state="normal")

    # Interfaccia utente
    num_particles_label = ctk.CTkLabel(window, text="Number of particles:")
    num_particles_entry = ctk.CTkEntry(window, textvariable=ctk.StringVar(value=num_particles), width=60)

    max_iterations_label = ctk.CTkLabel(window, text="Max iterations PSO:")
    max_iterations_entry = ctk.CTkEntry(window, textvariable=ctk.StringVar(value=max_iterations), width=60)

    w_label = ctk.CTkLabel(window, text="(Coefficient of inertia) w:")
    w_entry = ctk.CTkEntry(window, textvariable=ctk.StringVar(value=w), width=40)
    
    variable_w_checkbox = ctk.CTkCheckBox(window, text="Variable w", variable=variable_w)

    c1_label = ctk.CTkLabel(window, text="(Self best) c1:")
    c1_entry = ctk.CTkEntry(window, textvariable=ctk.StringVar(value=c1), width=40)

    c2_label = ctk.CTkLabel(window, text="(Global best) c2:")
    c2_entry = ctk.CTkEntry(window, textvariable=ctk.StringVar(value=c2), width=40)
    
    num_iterations_label = ctk.CTkLabel(window, text="# of iterations per problem:")
    num_iterations_entry = ctk.CTkEntry(window, textvariable=ctk.StringVar(value=num_iterations), width=40)

    run_button = ctk.CTkButton(window, text="Run study PSO", command=run_pso_study)
    
    progress_bar = ctk.CTkProgressBar(window, orientation="orizontal", width=300)
    progress_bar.set(0)

    result_label = ctk.CTkLabel(window, text="Results will be displayed here.")

    # Layout dell'interfaccia utente
    num_particles_label.grid(row=0, column=0, padx=10, pady=5, sticky=ctk.E)
    num_particles_entry.grid(row=0, column=1, padx=5, pady=5, sticky=ctk.W)

    max_iterations_label.grid(row=0, column=2, padx=10, pady=5, sticky=ctk.E)
    max_iterations_entry.grid(row=0, column=3, padx=5, pady=5, sticky=ctk.W)

    w_label.grid(row=2, column=0, padx=10, pady=5, sticky=ctk.E)
    w_entry.grid(row=2, column=1, padx=5, pady=5, sticky=ctk.W)
    
    variable_w_checkbox.grid(row=2, column=2, padx=5, pady=5, sticky=ctk.W)

    c1_label.grid(row=3, column=0, padx=10, pady=5, sticky=ctk.E)
    c1_entry.grid(row=3, column=1, padx=5, pady=5, sticky=ctk.W)

    c2_label.grid(row=3, column=2, padx=10, pady=5, sticky=ctk.E)
    c2_entry.grid(row=3, column=3, padx=5, pady=5, sticky=ctk.W)

    num_iterations_label.grid(row=4, column=0, padx=10, pady=5, sticky=ctk.E)
    num_iterations_entry.grid(row=4, column=1, padx=5, pady=5, sticky=ctk.W)

    run_button.grid(row=5, column=0, columnspan=4, pady=10)
    
    progress_bar.grid(row=6, column=0, columnspan=4, pady=10)

    result_label.grid(row=7, column=0, columnspan=4, pady=10)

    window.mainloop()


if __name__ == "__main__":
    # Parametri di default
    max_iterations = 5000
    num_particles = 20
    w, c1, c2, num_iterations = 0.7, 1.43, 1.43, 20

    # Avvia l'interfaccia utente
    study_gui()