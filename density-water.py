import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt

def calculate_radial_density(universe, group1_selection, group2_selection, bin_width=0.1):
    """
    Calculate the radial density of a specific group with respect to 
    the center of mass (COM) of another group.
    
    Parameters:
    - universe: MDAnalysis Universe object
    - group1_selection: Atom selection string for the group of interest
    - group2_selection: Atom selection string for the reference group
    - bin_width: Width of the radial bins (in the same unit as coordinates, typically Å)
    
    Returns:
    - bin_centers: Centers of the radial bins
    - density: Radial density values
    """
    group1 = universe.select_atoms(group1_selection)
    group2 = universe.select_atoms(group2_selection)
    
    # Initialize histogram
    max_distance = universe.dimensions[:3].min() / 2  # Assume periodic box
    bins = np.arange(0, max_distance, bin_width)
    bin_counts = np.zeros(len(bins) - 1)
    
    for ts in universe.trajectory:
        # Calculate the center of mass of the reference group
        com_group2 = group2.center_of_mass()
        
        # Calculate distances of all atoms in group1 to the COM of group2
        distances = np.linalg.norm(group1.positions - com_group2, axis=1)
        
        # Accumulate the histogram
        bin_counts += np.histogram(distances, bins=bins)[0]
    
    # Calculate density (normalized by shell volume)
    shell_volumes = (4 / 3) * np.pi * (bins[1:]**3 - bins[:-1]**3)
    density = bin_counts / (shell_volumes * len(universe.trajectory))
    
    # Return bin centers and density
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers, density

# Main function
if __name__ == "__main__":
    # Load the trajectory
    u = mda.Universe("md.pdb", "md-cntr.xtc")
    
    # Define groups (modify these selections based on your system)
    group1_sel = "resname TIP3 and name O*"  # Group whose density you want to calculate
    group2_sel = "resname ETHY"  # Reference group
    
    # Parameters
    bin_width = 0.1  # Bin width in Å
    
    # Calculate radial density
    bin_centers, density = calculate_radial_density(u, group1_sel, group2_sel, bin_width)
    # Plot the result
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, density, label="Radial Density")
    plt.xlabel("Distance from COM (Å)")
    plt.ylabel("Density (atoms/Å³)")
    plt.title("Radial Density Distribution")
    plt.legend()
    plt.grid()
    plt.show()
    # Save the results to a file
    output_filename = "radial_density-water.txt"
    data_to_save = np.column_stack((bin_centers, density))
    np.savetxt(output_filename, data_to_save, header="Bin_Center(Å) Density(atoms/Å³)", fmt="%.6f")
    print(f"Radial density data saved to {output_filename}")
