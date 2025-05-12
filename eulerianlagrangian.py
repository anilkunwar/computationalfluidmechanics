import streamlit as st
import yaml
import matplotlib.pyplot as plt
import numpy as np
import io

# Embedded YAML content (unchanged)
yaml_content = """
EulerianLagrangianFormulation:
  description: Hybrid computational approach for modeling particle-fluid interactions in multiphase flows, particularly in laser processing or combustion.
  frameworks:
    Eulerian:
      description: Models continuous phase (fluid, gas, plasma) on a fixed grid.
      governing_equations:
        - Continuity: "\\\\frac{\\\\partial \\\\rho_f}{\\\\partial t} + \\\\nabla \\\\cdot (\\\\rho_f \\\\mathbf{u}_f) = 0"
        - Navier-Stokes: "\\\\rho_f \\\\left( \\\\frac{\\\\partial \\\\mathbf{u}_f}{\\\\partial t} + (\\\\mathbf{u}_f \\\\cdot \\\\nabla) \\\\mathbf{u}_f \\\\right) = -\\\\nabla p + \\\\mu \\\\nabla^2 \\\\mathbf{u}_f + \\\\rho_f \\\\mathbf{g}"
      applications:
        - Gas flow in laser processing.
        - Air flow in combustion chambers.
        - Plasma dynamics in laser ablation.
    Lagrangian:
      description: Tracks discrete particles individually, solving for position, velocity, and other properties.
      governing_equations:
        - Motion: "m_p \\\\frac{d\\\\mathbf{v}_p}{dt} = \\\\mathbf{F}_{\\\\text{total}}, \\\\quad \\\\mathbf{F}_{\\\\text{total}} = \\\\mathbf{F}_{\\\\text{drag}} + \\\\mathbf{F}_{\\\\text{gravity}} + \\\\mathbf{F}_{\\\\text{other}}"
        - Energy: "m_p c_p \\\\frac{dT_p}{dt} = Q_{\\\\text{laser}} + Q_{\\\\text{conv}} + Q_{\\\\text{rad}} + Q_{\\\\text{phase}}"
      applications:
        - Powder particles in laser cladding.
        - Fuel droplets in air-fuel sprays.
        - Debris in laser ablation.
  particle_wall_interaction:
    description: Models behavior when particles collide with a wall (substrate, chamber wall).
    interaction_types:
      - ElasticCollision:
          description: Particle rebounds with energy loss determined by coefficient of restitution.
      - InelasticCollision:
          description: Particle loses significant energy, may stick to wall (e.g., in laser cladding).
      - ThermalEffects:
          description: Heat transfer between particle and heated/molten wall, including conduction and convection.
      - PhaseChange:
          description: Particle melting, vaporization, or solidification, modeled with latent heat and mass transfer.
    parameters:
      - CoefficientOfRestitution: Determines energy loss in collisions.
      - StickingProbability: Likelihood of particle adhesion to wall.
      - SplashingBehavior: For molten particles impacting a surface.
    laser_specific:
      - RecoilPressure: From material ejection in laser processing.
      - PlasmaInteraction: Particle interaction with laser-induced plasma.
  coupling_modes:
    - OneWay:
        description: Continuous phase affects particles, but particles do not influence continuous phase.
        use_case: Dilute particle systems.
    - TwoWay:
        description: Particles exchange momentum, energy, mass with continuous phase.
        use_case: Moderate particle concentrations in laser processing or sprays.
    - FourWay:
        description: Includes particle-particle collisions and particle-wall interactions.
        use_case: Dense particle systems (e.g., powder beds in selective laser melting).
  applications:
    - LaserCladding:
        description: Tracks powder particle deposition, melting, and adhesion to substrate.
    - AirFuelSpray:
        description: Models fuel droplet dispersion and interaction with air in combustion processes.
    - LaserAblation:
        description: Simulates debris ejection and redeposition on surface.
  advantages:
    - Detailed particle tracking (position, velocity, temperature).
    - Flexible for complex physics (laser-particle interactions, combustion).
    - Suitable for dilute and dense systems with appropriate coupling.
  challenges:
    - Computationally expensive for large particle counts or complex geometries.
    - Requires precise material properties and empirical coefficients.
    - Needs high-fidelity models for laser-particle and particle-fluid interactions.
"""

# Parse YAML content
data = yaml.safe_load(yaml_content)

# Initialize session state
if 'coeff_restitution' not in st.session_state:
    st.session_state.coeff_restitution = 0.8
if 'sticking_prob' not in st.session_state:
    st.session_state.sticking_prob = 0.5
if 'rerun_key' not in st.session_state:
    st.session_state.rerun_key = 0
if 'outcome_history' not in st.session_state:
    st.session_state.outcome_history = {'stick': 0, 'bounce': 0}

# Set Matplotlib parameters for publication quality
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.7
})

# Streamlit app
st.title("Eulerian-Lagrangian Formulation Explorer")
st.markdown("Explore the Eulerian-Lagrangian formulation for particle-fluid interactions in air-fuel sprays and laser cladding applications.")

# Sidebar for navigation
st.sidebar.header("Navigation")
section = st.sidebar.selectbox(
    "Select Section",
    ["Overview", "Frameworks", "Particle-Wall Interaction", "Coupling Modes", "Applications", "Parameter Configuration", "Visualization"]
)

# Overview section
if section == "Overview":
    st.header("Overview")
    st.write(data["EulerianLagrangianFormulation"]["description"])
    st.subheader("Advantages")
    for adv in data["EulerianLagrangianFormulation"]["advantages"]:
        st.write(f"- {adv}")
    st.subheader("Challenges")
    for chal in data["EulerianLagrangianFormulation"]["challenges"]:
        st.write(f"- {chal}")

# Frameworks section
elif section == "Frameworks":
    st.header("Frameworks")
    for framework, details in data["EulerianLagrangianFormulation"]["frameworks"].items():
        st.subheader(framework)
        st.write(details["description"])
        st.write("**Governing Equations:**")
        for eq in details["governing_equations"]:
            for name, eq_text in eq.items():
                st.markdown(f"- **{name}**:")
                st.latex(eq_text)
        st.write("**Applications:**")
        for app in details["applications"]:
            st.write(f"- {app}")

# Particle-Wall Interaction section
elif section == "Particle-Wall Interaction":
    st.header("Particle-Wall Interaction")
    st.write(data["EulerianLagrangianFormulation"]["particle_wall_interaction"]["description"])
    st.subheader("Interaction Types")
    for interaction in data["EulerianLagrangianFormulation"]["particle_wall_interaction"]["interaction_types"]:
        for name, details in interaction.items():
            st.markdown(f"- **{name}**: {details['description']}")
    st.subheader("Parameters")
    for param in data["EulerianLagrangianFormulation"]["particle_wall_interaction"]["parameters"]:
        st.write(f"- {param}")
    st.subheader("Laser-Specific Effects")
    for effect in data["EulerianLagrangianFormulation"]["particle_wall_interaction"]["laser_specific"]:
        st.write(f"- {effect}")
    st.subheader("Governing Equations")
    st.latex(r"""
    \text{Elastic Collision: } v_{p,y}(t^+) = -e v_{p,y}(t^-), \quad e = \text{Coefficient of Restitution}
    """)
    st.latex(r"""
    \text{Sticking (Laser Cladding): Particle sticks with probability } P_{\text{stick}}
    """)

# Coupling Modes section
elif section == "Coupling Modes":
    st.header("Coupling Modes")
    for mode in data["EulerianLagrangianFormulation"]["coupling_modes"]:
        for name, details in mode.items():
            st.subheader(name)
            st.write(f"**Description**: {details['description']}")
            st.write(f"**Use Case**: {details['use_case']}")

# Applications section
elif section == "Applications":
    st.header("Applications")
    for app in data["EulerianLagrangianFormulation"]["applications"]:
        for name, details in app.items():
            st.subheader(name)
            st.write(details["description"])

# Parameter Configuration section
elif section == "Parameter Configuration":
    st.header("Configure Particle-Wall Interaction Parameters")
    st.write("""
    Set parameters for particle-wall interactions in the Visualization section:
    - **Coefficient of Restitution (e)**: Controls bounce energy loss in both Air-Fuel Interaction (elastic collisions) and Laser Cladding (bounces when not sticking).
    - **Sticking Probability (P_stick)**: Controls the likelihood of the particle sticking to the substrate in Laser Cladding only.
    Changes are applied immediately to simulations in the Visualization section.
    """)
    st.session_state.coeff_restitution = st.slider("Coefficient of Restitution", 0.0, 1.0, st.session_state.coeff_restitution)
    st.session_state.sticking_prob = st.slider("Sticking Probability", 0.0, 1.0, st.session_state.sticking_prob)
    st.write("**Configured Parameters:**")
    st.write(f"- Coefficient of Restitution: {st.session_state.coeff_restitution}")
    st.write(f"- Sticking Probability: {st.session_state.sticking_prob} (Used in Laser Cladding for adhesion)")
    if st.button("Save Configuration"):
        config = {
            "CoefficientOfRestitution": st.session_state.coeff_restitution,
            "StickingProbability": st.session_state.sticking_prob
        }
        st.download_button(
            label="Download Configuration",
            data=yaml.dump(config),
            file_name="particle_wall_config.yaml",
            mime="text/yaml"
        )

# Visualization section
elif section == "Visualization":
    st.header("Eulerian-Lagrangian Visualization")
    st.write("""
    Compare particle-fluid interactions in two interfaces: Air-Fuel Interaction (e.g., fuel spray in combustion) and 
    Laser Cladding (e.g., powder deposition in additive manufacturing). Select a mode to visualize tailored fluid fields 
    and particle behaviors, with interactive parameters.
    """)

    # General Governing Equations
    st.subheader("General Governing Equations")
    st.markdown("These equations describe the Eulerian, Lagrangian, and Eulerian-Lagrangian frameworks used in the simulations.")
    st.latex(r"""
    \textbf{Eulerian Framework (Continuous Phase):}
    \begin{align}
    \frac{\partial \rho_f}{\partial t} + \nabla \cdot (\rho_f \mathbf{u}_f) &= 0 \tag{1} \\
    \rho_f \left( \frac{\partial \mathbf{u}_f}{\partial t} + (\mathbf{u}_f \cdot \nabla) \mathbf{u}_f \right) &= -\nabla p + \mu \nabla^2 \mathbf{u}_f + \rho_f \mathbf{g} \tag{2}
    \end{align}
    """)
    st.latex(r"""
    \textbf{Lagrangian Framework (Discrete Particles):}
    \begin{align}
    m_p \frac{d\mathbf{v}_p}{dt} &= \mathbf{F}_{\text{total}}, & \mathbf{F}_{\text{total}} &= \mathbf{F}_{\text{drag}} + \mathbf{F}_{\text{gravity}} + \mathbf{F}_{\text{other}} \tag{1} \\
    m_p c_p \frac{dT_p}{dt} &= Q_{\text{laser}} + Q_{\text{conv}} + Q_{\text{rad}} + Q_{\text{phase}} \tag{2}
    \end{align}
    """)
    st.latex(r"""
    \textbf{Eulerian-Lagrangian Framework (Coupled System):}
    \begin{align}
    \text{Fluid (Eulerian):} \quad \frac{\partial \rho_f}{\partial t} + \nabla \cdot (\rho_f \mathbf{u}_f) &= 0 \tag{1} \\
    \rho_f \left( \frac{\partial \mathbf{u}_f}{\partial t} + (\mathbf{u}_f \cdot \nabla) \mathbf{u}_f \right) &= -\nabla p + \mu \nabla^2 \mathbf{u}_f + \rho_f \mathbf{g} \tag{2} \\
    \text{Particles (Lagrangian):} \quad m_p \frac{d\mathbf{v}_p}{dt} &= \mathbf{F}_{\text{drag}} + \mathbf{F}_{\text{gravity}} + \mathbf{F}_{\text{other}}, \quad \mathbf{F}_{\text{drag}} = 0.5 \rho_f C_d A_p |\mathbf{v}_{\text{rel}}| \mathbf{v}_{\text{rel}} \tag{3} \\
    \text{Coupling (One-Way):} \quad \mathbf{v}_{\text{rel}} &= \mathbf{v}_p - \mathbf{u}_f(\mathbf{x}_p, t) \tag{4}
    \end{align}
    """)

    # Mode selection
    mode = st.selectbox("Select Interface", ["Air-Fuel Interaction", "Laser Cladding"])

    # Application-Specific Governing Equations
    st.subheader("Application-Specific Governing Equations")
    if mode == "Air-Fuel Interaction":
        st.latex(r"""
        \textbf{Governing Equations for Air-Fuel Interaction:}
        \begin{align}
        m_p \frac{d\mathbf{v}_p}{dt} &= \mathbf{F}_{\text{total}}, & \mathbf{F}_{\text{total}} &= \mathbf{F}_{\text{drag}} + \mathbf{F}_{\text{gravity}} \tag{1} \\
        \mathbf{F}_{\text{drag}} &= 0.5 \rho C_d A_p |\mathbf{v}_{\text{rel}}| \mathbf{v}_{\text{rel}}, & \mathbf{v}_{\text{rel}} &= \mathbf{v}_p - \mathbf{u}_f \tag{2} \\
        \mathbf{F}_{\text{gravity}} &= m_p \mathbf{g}, & \mathbf{g} &= [0, -g] \tag{3} \\
        \text{If collision at } y = 0: \quad v_{p,y}(t^+) &= -e v_{p,y}(t^-) \tag{4}
        \end{align}
        """)
    else:
        st.latex(r"""
        \textbf{Governing Equations for Laser Cladding:}
        \begin{align}
        m_p \frac{d\mathbf{v}_p}{dt} &= \mathbf{F}_{\text{total}}, & \mathbf{F}_{\text{total}} &= \mathbf{F}_{\text{drag}} + \mathbf{F}_{\text{gravity}} + \mathbf{F}_{\text{recoil}} \tag{1} \\
        \mathbf{F}_{\text{drag}} &= 0.5 \rho C_d A_p |\mathbf{v}_{\text{rel}}| \mathbf{v}_{\text{rel}}, & \mathbf{v}_{\text{rel}} &= \mathbf{v}_p - \mathbf{u}_f \tag{2} \\
        \mathbf{F}_{\text{gravity}} &= m_p \mathbf{g}, & \mathbf{g} &= [0, -g] \tag{3} \\
        \mathbf{F}_{\text{recoil}} &= [0, F_{\text{recoil}}] \quad \text{for } t < 1.0 \, \text{s}, & \text{else } \mathbf{F}_{\text{recoil}} &= [0, 0] \tag{4} \\
        \text{If collision at } y = 0: \quad v_{p,y}(t^+) &= -e v_{p,y}(t^-) \quad \text{or stick with probability } P_{\text{stick}} \tag{5}
        \end{align}
        """)

    # Common simulation parameters
    dt = 0.001  # Time step (reduced for stability)
    t_max = 2.0  # Simulation time
    t = np.arange(0, t_max, dt)
    m_p = 1e-6  # Particle mass (kg)
    rho = 1.2  # Fluid density (kg/m^3)
    d_p = 5e-4  # Particle diameter (m)
    C_d = 0.47  # Drag coefficient
    A_p = np.pi * (d_p / 2) ** 2  # Particle cross-sectional area
    g = 9.81  # Gravity (m/s^2)
    wall_y = 0.0  # Wall at y=0
    v_rel_max = 50.0  # Maximum relative velocity (m/s)

    # Air-Fuel Interaction Mode
    if mode == "Air-Fuel Interaction":
        st.subheader("Air-Fuel Interaction")
        st.write("""
        Models fuel droplets (Lagrangian) in a turbulent air flow (Eulerian), as in a combustion chamber or fuel spray. 
        The fluid field is a turbulent jet, and multiple particles disperse due to drag, with elastic collisions at the wall.
        """)
        with st.expander("Learn More"):
            st.markdown("""
            In air-fuel interactions, air flow governs fuel droplet motion via drag, affecting spray patterns and combustion efficiency. 
            The turbulent fluid field mimics real-world mixing in engines or burners. Particles bounce elastically at the wall (y=0) with a coefficient of restitution.
            """)
            st.latex(r"m_p \frac{d\mathbf{v}_p}{dt} = \mathbf{F}_{\text{drag}} + \mathbf{F}_{\text{gravity}}")
            st.latex(r"\mathbf{F}_{\text{drag}} = 0.5 \rho C_d A_p |\mathbf{v}_{\text{rel}}| \mathbf{v}_{\text{rel}}")
            st.latex(r"\mathbf{v}_{\text{rel}} = \mathbf{v}_p - \mathbf{u}_f")
            st.latex(r"\text{Collision: } v_{p,y}(t^+) = -e v_{p,y}(t^-)")

        # User inputs
        jet_strength = st.slider("Jet Strength (m/s)", 0.0, 5.0, 3.0)
        turbulence_intensity = st.slider("Turbulence Intensity", 0.0, 0.5, 0.2)
        num_particles = st.slider("Number of Particles", 1, 5, 3)
        compare_e = st.checkbox("Compare Different Coefficients of Restitution", value=False)
        show_collisions = st.checkbox("Show Collision Markers", value=True)

        # Fluid field
        x = np.linspace(-2, 2, 10)
        y = np.linspace(-1, 6, 10)
        X, Y = np.meshgrid(x, y)
        U = jet_strength * np.exp(-((X**2 + (Y-3)**2) / 0.5))
        V = np.zeros_like(Y)
        U += turbulence_intensity * np.random.normal(0, jet_strength, U.shape)
        V += turbulence_intensity * np.random.normal(0, jet_strength, V.shape)

        # Initialize particles
        e_values = [0.2, 0.5, 0.8] if compare_e else [st.session_state.coeff_restitution]
        pos_all = {}
        vel_all = {}
        collisions_all = {}

        for e in e_values:
            pos = np.zeros((len(t), num_particles, 2))
            vel = np.zeros((len(t), num_particles, 2))
            drag_force = np.zeros((len(t), num_particles, 2))
            collisions = [[] for _ in range(num_particles)]
            for p in range(num_particles):
                pos[0, p] = [np.random.uniform(-0.5, 0.5), 5.0]
                vel[0, p] = [np.random.uniform(1.5, 2.5), -2.0]

            # Simulate
            for i in range(1, len(t)):
                for p in range(num_particles):
                    u_f_x = jet_strength * np.exp(-((pos[i-1, p, 0]**2 + (pos[i-1, p, 1]-3)**2) / 0.5))
                    u_f_y = 0.0
                    u_f = np.array([u_f_x, u_f_y]) + turbulence_intensity * np.random.normal(0, jet_strength, 2)
                    v_rel = vel[i-1, p] - u_f
                    v_rel_mag = np.sqrt(v_rel[0]**2 + v_rel[1]**2)
                    v_rel_mag = min(v_rel_mag, v_rel_max)  # Cap relative velocity
                    if v_rel_mag > 0:
                        v_rel_unit = v_rel / v_rel_mag
                    else:
                        v_rel_unit = np.zeros(2)
                    F_drag = 0.5 * rho * C_d * A_p * v_rel_mag * (v_rel_mag * v_rel_unit)
                    drag_force[i, p] = F_drag
                    F_gravity = np.array([0.0, -m_p * g])
                    F_total = F_drag + F_gravity
                    acc = F_total / m_p
                    vel[i, p] = vel[i-1, p] + acc * dt
                    pos[i, p] = pos[i-1, p] + vel[i-1, p] * dt
                    if pos[i, p, 1] <= wall_y and vel[i-1, p, 1] < 0:
                        vel[i, p, 1] = -e * vel[i-1, p, 1]
                        pos[i, p, 1] = wall_y
                        collisions[p].append((t[i], pos[i, p, 0], pos[i, p, 1]))

            pos_all[e] = pos
            vel_all[e] = vel
            collisions_all[e] = collisions

        # Plot 1: Fluid field with trajectories
        fig1 = plt.figure(figsize=(6, 4))
        ax1 = fig1.add_subplot(111)
        ax1.quiver(X, Y, U, V, color='blue', label='Air Velocity')
        colors = ['red', 'darkorange', 'purple']
        for idx, e in enumerate(e_values):
            for p in range(num_particles):
                label = f'Fuel Droplet (e={e})' if p == 0 else ""
                ax1.plot(pos_all[e][:, p, 0], pos_all[e][:, p, 1], color=colors[idx % len(colors)], alpha=0.6, label=label)
                if show_collisions:
                    for t_coll, x_coll, y_coll in collisions_all[e][p]:
                        ax1.plot(x_coll, y_coll, 'bo', markersize=8, label='Collision' if p == 0 and idx == 0 else "")
                        ax1.annotate(f'e={e}', (x_coll, y_coll + 0.2), fontsize=8, color=colors[idx % len(colors)])
        ax1.axhline(wall_y, color='gray', linestyle='--', label='Wall')
        ax1.annotate('Turbulent Jet', xy=(0, 3), xytext=(1, 4), arrowprops=dict(facecolor='black', shrink=0.05))
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Air-Fuel Interaction')
        ax1.legend()
        ax1.grid(True)
        ax1.text(0.05, 0.95, f'e={e_values}', transform=ax1.transAxes, fontsize=10, verticalalignment='top')
        fig1.tight_layout()
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png', dpi=300)
        st.image(buf1)

        # Plot 2: Y-velocity
        st.write("Y-velocity of the first particle over time, showing elastic collisions at the wall.")
        fig2 = plt.figure(figsize=(6, 4))
        ax2 = fig2.add_subplot(111)
        for idx, e in enumerate(e_values):
            ax2.plot(t, vel_all[e][:, 0, 1], color=colors[idx % len(colors)], label=f'Y-Velocity (e={e})')
            if show_collisions:
                for t_coll, _, _ in collisions_all[e][0]:
                    ax2.axvline(t_coll, color=colors[idx % len(colors)], linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Y-Velocity (m/s)')
        ax2.set_title('Particle-Wall Interaction (First Particle)')
        ax2.legend()
        ax2.grid(True)
        fig2.tight_layout()
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png', dpi=300)
        st.image(buf2)

    # Laser Cladding Mode
    else:
        st.subheader("Laser Cladding")
        st.write("""
        Models a powder particle (Lagrangian) in a carrier gas jet (Eulerian), directed toward a substrate under laser heating. 
        Includes recoil pressure and probabilistic sticking or elastic bouncing at the substrate. Use the 'Rerun Simulation' button to see different sticking/bouncing outcomes based on the Sticking Probability.
        Adjust the Carrier Gas Jet Strength to see changes in lateral dispersion and the Recoil Pressure Strength to alter the particle's vertical motion.
        """)
        with st.expander("Learn More"):
            st.markdown("""
            In laser cladding, a carrier gas delivers powder particles to a melt pool, where laser-induced recoil pressure and 
            particle-wall interactions (sticking or bouncing) determine deposition quality.
            """)
            st.latex(r"m_p \frac{d\mathbf{v}_p}{dt} = \mathbf{F}_{\text{total}}")
            st.latex(r"\mathbf{F}_{\text{total}} = \mathbf{F}_{\text{drag}} + \mathbf{F}_{\text{gravity}} + \mathbf{F}_{\text{recoil}}")
            st.latex(r"\mathbf{F}_{\text{drag}} = 0.5 \rho C_d A_p |\mathbf{v}_{\text{rel}}| \mathbf{v}_{\text{rel}}")
            st.latex(r"\mathbf{v}_{\text{rel}} = \mathbf{v}_p - \mathbf{u}_f")
            st.latex(r"\text{If collision: } v_{p,y}(t^+) = -e v_{p,y}(t^-) \text{ or stick with probability } P_{\text{stick}}")

        # User inputs
        jet_strength = st.slider("Carrier Gas Jet Strength (m/s)", 0.0, 10.0, 5.0)
        recoil_strength = st.slider("Recoil Pressure Strength (N)", 0.0, 5e-5, 1e-5, step=1e-6)
        st.write(f"Selected Recoil Strength: {recoil_strength} N")
        coeff_restitution = st.session_state.coeff_restitution
        sticking_prob = st.session_state.sticking_prob
        show_collisions = st.checkbox("Show Collision Markers", value=True)
        if st.button("Rerun Simulation"):
            st.session_state.rerun_key += 1
            np.random.seed(st.session_state.rerun_key)
        st.write(f"Using Coefficient of Restitution: {coeff_restitution}, Sticking Probability: {sticking_prob}")
        st.write(f"Outcome History: Sticking = {st.session_state.outcome_history['stick']}, Bouncing = {st.session_state.outcome_history['bounce']}")

        # Fluid field
        x = np.linspace(-2, 2, 10)
        y = np.linspace(-1, 6, 10)
        X, Y = np.meshgrid(x, y)
        U = jet_strength * np.exp(-((X**2 + (Y-3)**2) / 0.5))
        V = np.zeros_like(Y)

        # Initialize particle
        np.random.seed(st.session_state.rerun_key)
        pos = np.zeros((len(t), 2))
        vel = np.zeros((len(t), 2))
        drag_force = np.zeros((len(t), 2))
        pos[0] = [np.random.uniform(-0.1, 0.1), 5.0]
        vel[0] = [0.5, -1.0]  # More realistic initial velocity
        stuck = False
        stick_time = None
        collisions = []
        outcome = None
        max_v_rel = 0.0
        max_F_drag = 0.0

        # Simulate
        for i in range(1, len(t)):
            if stuck:
                pos[i] = pos[i-1]
                vel[i] = [0.0, 0.0]
                drag_force[i] = [0.0, 0.0]  # Zero drag force after sticking
                continue

            u_f_x = jet_strength * np.exp(-((pos[i-1, 0]**2 + (pos[i-1, 1]-3)**2) / 0.5))
            u_f_y = 0.0
            u_f = np.array([u_f_x, u_f_y])
            v_rel = vel[i-1] - u_f
            v_rel_mag = np.sqrt(v_rel[0]**2 + v_rel[1]**2)
            max_v_rel = max(max_v_rel, v_rel_mag)
            v_rel_mag = min(v_rel_mag, v_rel_max)  # Cap relative velocity
            if v_rel_mag > 0:
                v_rel_unit = v_rel / v_rel_mag
            else:
                v_rel_unit = np.zeros(2)
            F_drag = 0.5 * rho * C_d * A_p * v_rel_mag * (v_rel_mag * v_rel_unit)
            if not np.isfinite(F_drag).all():
                F_drag = np.zeros(2)  # Prevent overflow
            drag_force[i] = F_drag
            max_F_drag = max(max_F_drag, np.sqrt(F_drag[0]**2 + F_drag[1]**2))
            F_gravity = np.array([0.0, -m_p * g])
            F_recoil = np.array([0.0, recoil_strength]) if t[i] < 1.0 else np.array([0.0, 0.0])
            F_total = F_drag + F_gravity + F_recoil
            acc = F_total / m_p
            vel[i] = vel[i-1] + acc * dt
            pos[i] = pos[i-1] + vel[i-1] * dt

            if pos[i, 1] <= wall_y and vel[i-1, 1] < 0:
                if np.random.random() < sticking_prob:
                    stuck = True
                    stick_time = t[i]
                    pos[i, 1] = wall_y
                    collisions.append((t[i], pos[i, 0], pos[i, 1], 'stick'))
                    outcome = 'stick'
                else:
                    vel[i, 1] = -coeff_restitution * vel[i-1, 1]
                    pos[i, 1] = wall_y
                    collisions.append((t[i], pos[i, 0], pos[i, 1], 'bounce'))
                    outcome = 'bounce'

        # Debug output (remove in production)
        st.write(f"Debug: Max |v_rel| = {max_v_rel:.2f} m/s, Max |F_drag| = {max_F_drag:.2e} N")

        # Update outcome history
        if outcome:
            st.session_state.outcome_history[outcome] += 1
            st.write(f"Current Run Outcome: {outcome.capitalize()}")

        # Plot 1: Fluid field with trajectory
        fig1 = plt.figure(figsize=(6, 4))
        ax1 = fig1.add_subplot(111)
        ax1.quiver(X, Y, U, V, color='blue', label='Carrier Gas')
        ax1.plot(pos[:, 0], pos[:, 1], 'r-', label='Particle Trajectory')
        ax1.axhline(wall_y, color='gray', linestyle='--', label='Substrate')
        if show_collisions:
            for t_coll, x_coll, y_coll, coll_type in collisions:
                if coll_type == 'stick':
                    ax1.plot(x_coll, y_coll, 'r*', markersize=12, label='Sticking Point')
                    ax1.annotate(f'Stick, P={sticking_prob}', (x_coll, y_coll + 0.2), fontsize=8, color='red')
                else:
                    ax1.plot(x_coll, y_coll, 'bo', markersize=8, label='Bounce' if collisions.index((t_coll, x_coll, y_coll, coll_type)) == 0 else "")
                    ax1.annotate(f'e={coeff_restitution}', (x_coll, y_coll + 0.2), fontsize=8, color='blue')
        ax1.annotate('Carrier Gas Jet', xy=(0, 3), xytext=(1, 4), arrowprops=dict(facecolor='black', shrink=0.05))
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Laser Cladding')
        ax1.legend()
        ax1.grid(True)
        ax1.text(0.05, 0.95, f'e={coeff_restitution}, P_stick={sticking_prob}, Jet={jet_strength} m/s', transform=ax1.transAxes, fontsize=10, verticalalignment='top')
        fig1.tight_layout()
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png', dpi=300)
        st.image(buf1)

        # Plot 2: Y-velocity
        st.write("Y-velocity of the particle over time, showing collisions or sticking at the substrate.")
        fig2 = plt.figure(figsize=(6, 4))
        ax2 = fig2.add_subplot(111)
        ax2.plot(t, vel[:, 1], color='darkgreen', label='Y-Velocity')
        if show_collisions:
            for t_coll, _, _, coll_type in collisions:
                ax2.axvline(t_coll, color='red' if coll_type == 'stick' else 'blue', linestyle='--', label=coll_type.capitalize() + ' Event')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Y-Velocity (m/s)')
        ax2.set_title('Particle-Wall Interaction')
        ax2.legend()
        ax2.grid(True)
        fig2.tight_layout()
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png', dpi=300)
        st.image(buf2)

        # Plot 3: Drag force magnitude
        st.write("Drag force magnitude over time, showing the influence of the carrier gas jet.")
        fig3 = plt.figure(figsize=(6, 4))
        ax3 = fig3.add_subplot(111)
        drag_mag = np.sqrt(drag_force[:, 0]**2 + drag_force[:, 1]**2)
        ax3.plot(t, drag_mag, color='purple', label='Drag Force')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Drag Force (N)')
        ax3.set_title('Drag Force Magnitude')
        ax3.legend()
        ax3.grid(True)
        fig3.tight_layout()
        buf3 = io.BytesIO()
        fig3.savefig(buf3, format='png', dpi=300)
        st.image(buf3)

# Footer
st.markdown("---")
st.markdown("Eulerian and Lagrangian Methods in Computational Fluid Mechanics.")
