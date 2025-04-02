import numpy as np
import re
import time
from sympy import symbols, expand
from typing import List, Dict, Tuple, Union, Optional

class MCPQuantumSimulator:
    """
    Master Control Program (MCP) Quantum Computer Simulator
    A natural language interface for simulating quantum computing operations
    """
    def __init__(self, qubits: int = 5):
        """Initialize the quantum simulator with a specified number of qubits"""
        self.num_qubits = qubits
        self.reset_state()
        
        # Define basic quantum gates
        self.gates = {
            "H": self._hadamard_gate,
            "X": self._pauli_x_gate,
            "Y": self._pauli_y_gate,
            "Z": self._pauli_z_gate,
            "CNOT": self._cnot_gate,
            "CZ": self._cz_gate,
            "T": self._t_gate,
            "S": self._s_gate
        }
        
        # Command parsing patterns
        self.command_patterns = [
            (r"(?i)add (\d+) qubits?", self._add_qubits),
            (r"(?i)reset( system)?", self.reset_state),
            (r"(?i)apply ([H|X|Y|Z|T|S]+) (gate )?to qubit (\d+)", self._apply_single_gate),
            (r"(?i)apply CNOT (gate )?with control (\d+) and target (\d+)", self._apply_cnot),
            (r"(?i)apply CZ (gate )?with control (\d+) and target (\d+)", self._apply_cz),
            (r"(?i)measure( all)?", self._measure_all),
            (r"(?i)measure qubit (\d+)", self._measure_qubit),
            (r"(?i)run ([a-zA-Z0-9_]+) algorithm", self._run_algorithm),
            (r"(?i)simulate random circuit sampling", self._simulate_rcs),
            (r"(?i)show (current )?state", self._show_state),
            (r"(?i)how many qubits", self._show_num_qubits)
        ]
    
    def reset_state(self) -> str:
        """Reset the quantum system to |0⟩^⊗n state"""
        # Initialize to |0⟩^⊗n state
        self.state = np.zeros(2**self.num_qubits, dtype=complex)
        self.state[0] = 1.0
        return "System reset to |0⟩^⊗n state"
    
    def process_command(self, command: str) -> str:
        """Process a natural language command for the quantum simulator"""
        for pattern, handler in self.command_patterns:
            match = re.match(pattern, command)
            if match:
                return handler(*match.groups())
        
        # If no pattern matched, try to understand as a custom circuit
        if "circuit" in command.lower():
            return "Please specify the circuit operations in sequence using 'apply' commands"
        
        return f"Command not recognized. Try something like:\n" + \
               f"- 'add 2 qubits'\n" + \
               f"- 'apply H to qubit 0'\n" + \
               f"- 'apply CNOT with control 0 and target 1'\n" + \
               f"- 'measure'\n" + \
               f"- 'reset system'"
    
    def _add_qubits(self, n: str) -> str:
        """Add n qubits to the system"""
        n = int(n)
        old_num_qubits = self.num_qubits
        
        # Save the current state
        old_state = self.state.copy()
        
        # Update the number of qubits
        self.num_qubits += n
        
        # Create a new state with additional qubits initialized to |0⟩
        self.state = np.zeros(2**self.num_qubits, dtype=complex)
        
        # Copy the old state amplitudes to the new state
        for i in range(len(old_state)):
            self.state[i] = old_state[i]
        
        return f"Added {n} qubits. System now has {self.num_qubits} qubits."
    
    def _show_num_qubits(self) -> str:
        """Return the current number of qubits"""
        return f"The system currently has {self.num_qubits} qubits"
    
    def _hadamard_gate(self, qubit: int) -> None:
        """Apply Hadamard gate to a single qubit"""
        h_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        self._apply_gate(h_gate, qubit)
    
    def _pauli_x_gate(self, qubit: int) -> None:
        """Apply Pauli X gate (NOT) to a single qubit"""
        x_gate = np.array([[0, 1], [1, 0]])
        self._apply_gate(x_gate, qubit)
    
    def _pauli_y_gate(self, qubit: int) -> None:
        """Apply Pauli Y gate to a single qubit"""
        y_gate = np.array([[0, -1j], [1j, 0]])
        self._apply_gate(y_gate, qubit)
    
    def _pauli_z_gate(self, qubit: int) -> None:
        """Apply Pauli Z gate to a single qubit"""
        z_gate = np.array([[1, 0], [0, -1]])
        self._apply_gate(z_gate, qubit)
    
    def _t_gate(self, qubit: int) -> None:
        """Apply T gate to a single qubit"""
        t_gate = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
        self._apply_gate(t_gate, qubit)
    
    def _s_gate(self, qubit: int) -> None:
        """Apply S gate to a single qubit"""
        s_gate = np.array([[1, 0], [0, 1j]])
        self._apply_gate(s_gate, qubit)
    
    def _cnot_gate(self, control: int, target: int) -> None:
        """Apply CNOT gate with given control and target qubits"""
        dim = 2**self.num_qubits
        cnot = np.eye(dim, dtype=complex)
        
        # For each computational basis state
        for i in range(dim):
            # Check if control qubit is 1
            if (i >> control) & 1:
                # Flip the target qubit
                target_mask = 1 << target
                flipped_i = i ^ target_mask
                
                # Swap the amplitudes
                cnot[i, i] = 0
                cnot[flipped_i, i] = 1
        
        self.state = cnot @ self.state
    
    def _cz_gate(self, control: int, target: int) -> None:
        """Apply CZ gate with given control and target qubits"""
        # Apply phase flip (-1) only when both qubits are |1⟩
        for i in range(2**self.num_qubits):
            if ((i >> control) & 1) and ((i >> target) & 1):
                self.state[i] *= -1
    
    def _apply_gate(self, gate: np.ndarray, qubit: int) -> None:
        """Apply a single-qubit gate to the specified qubit"""
        # Create identity matrices for other qubits
        op = np.array([[1]])
        
        for q in range(self.num_qubits):
            if q == qubit:
                op = np.kron(op, gate)
            else:
                op = np.kron(op, np.eye(2))
        
        self.state = op @ self.state
    
    def _apply_single_gate(self, gate: str, _, qubit: str) -> str:
        """Apply a single-qubit gate specified in natural language"""
        qubit = int(qubit)
        if qubit >= self.num_qubits:
            return f"Error: Qubit {qubit} does not exist. System has {self.num_qubits} qubits (indexed 0 to {self.num_qubits-1})."
        
        gate = gate.upper()
        if gate in self.gates:
            self.gates[gate](qubit)
            return f"Applied {gate} gate to qubit {qubit}"
        else:
            return f"Gate {gate} not recognized. Available gates: {', '.join(self.gates.keys())}"
    
    def _apply_cnot(self, _, control: str, target: str) -> str:
        """Apply CNOT gate with control and target specified in natural language"""
        control, target = int(control), int(target)
        
        if control >= self.num_qubits or target >= self.num_qubits:
            return f"Error: Qubits must be in range 0-{self.num_qubits-1}"
        
        if control == target:
            return "Error: Control and target qubits must be different"
        
        self._cnot_gate(control, target)
        return f"Applied CNOT gate with control={control} and target={target}"
    
    def _apply_cz(self, _, control: str, target: str) -> str:
        """Apply CZ gate with control and target specified in natural language"""
        control, target = int(control), int(target)
        
        if control >= self.num_qubits or target >= self.num_qubits:
            return f"Error: Qubits must be in range 0-{self.num_qubits-1}"
        
        if control == target:
            return "Error: Control and target qubits must be different"
        
        self._cz_gate(control, target)
        return f"Applied CZ gate with control={control} and target={target}"
    
    def _measure_all(self, _=None) -> str:
        """Perform a measurement of all qubits"""
        # Calculate probabilities
        probabilities = np.abs(self.state)**2
        
        # Simulate measurement
        result = np.random.choice(2**self.num_qubits, p=probabilities)
        
        # Convert to binary string
        binary_result = format(result, f'0{self.num_qubits}b')
        
        # Collapse state
        self.state = np.zeros(2**self.num_qubits, dtype=complex)
        self.state[result] = 1.0
        
        return f"Measurement result: |{binary_result}⟩"
    
    def _measure_qubit(self, qubit: str) -> str:
        """Measure a specific qubit"""
        qubit = int(qubit)
        if qubit >= self.num_qubits:
            return f"Error: Qubit {qubit} does not exist"
        
        # Calculate probabilities for the qubit being 0 or 1
        prob_0 = 0
        prob_1 = 0
        
        for i in range(2**self.num_qubits):
            if (i >> qubit) & 1:  # If qubit is 1
                prob_1 += np.abs(self.state[i])**2
            else:  # If qubit is 0
                prob_0 += np.abs(self.state[i])**2
        
        # Normalize probabilities
        total_prob = prob_0 + prob_1
        if total_prob > 0:
            prob_0 /= total_prob
            prob_1 /= total_prob
        
        # Simulate measurement
        result = np.random.choice([0, 1], p=[prob_0, prob_1])
        
        # Collapse state
        new_state = np.zeros(2**self.num_qubits, dtype=complex)
        norm_factor = 0
        
        for i in range(2**self.num_qubits):
            bit_val = (i >> qubit) & 1
            if bit_val == result:
                new_state[i] = self.state[i]
                norm_factor += np.abs(self.state[i])**2
        
        # Normalize
        if norm_factor > 0:
            new_state /= np.sqrt(norm_factor)
        
        self.state = new_state
        
        return f"Qubit {qubit} measured: |{result}⟩"
    
    def _run_algorithm(self, algorithm: str) -> str:
        """Run a predefined quantum algorithm"""
        algorithm = algorithm.lower()
        
        if algorithm == "grover":
            return self._run_grover_algorithm()
        elif algorithm == "deutsch_jozsa":
            return self._run_deutsch_jozsa_algorithm()
        elif algorithm == "bell":
            return self._create_bell_state()
        elif algorithm == "teleportation":
            return self._run_teleportation()
        elif algorithm == "shor":
            return "Shor's algorithm is not implemented in this simulator as it requires too many qubits for classical simulation"
        elif algorithm == "qft":
            return self._run_qft()
        else:
            return f"Algorithm '{algorithm}' not recognized. Available algorithms: grover, deutsch_jozsa, bell, teleportation, qft"
    
    def _create_bell_state(self) -> str:
        """Create a Bell state between qubits 0 and 1"""
        if self.num_qubits < 2:
            return "Need at least 2 qubits for Bell state"
        
        # Reset to |0⟩ state
        self.reset_state()
        
        # Apply Hadamard to qubit 0
        self._hadamard_gate(0)
        
        # Apply CNOT with control 0 and target 1
        self._cnot_gate(0, 1)
        
        return "Created Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 between qubits 0 and 1"
    
    def _run_deutsch_jozsa_algorithm(self) -> str:
        """Implement a simple Deutsch-Jozsa algorithm"""
        if self.num_qubits < 2:
            return "Need at least 2 qubits for Deutsch-Jozsa algorithm"
        
        # Reset to |0⟩ state
        self.reset_state()
        
        # Prepare input: |1⟩ for ancilla qubit
        self._pauli_x_gate(self.num_qubits - 1)
        
        # Apply Hadamard to all qubits
        for i in range(self.num_qubits):
            self._hadamard_gate(i)
        
        # For demonstration, implement a balanced oracle using CNOT gates
        for i in range(self.num_qubits - 1):
            self._cnot_gate(i, self.num_qubits - 1)
        
        # Apply Hadamard to all qubits except the ancilla
        for i in range(self.num_qubits - 1):
            self._hadamard_gate(i)
        
        # Measure qubits 0 to n-2
        result = ""
        for i in range(self.num_qubits - 1):
            result += self._measure_qubit(i) + "\n"
        
        return f"Deutsch-Jozsa algorithm completed.\nResults:\n{result}\nFunction is balanced."
    
    def _run_grover_algorithm(self) -> str:
        """Implement a simplified Grover's search algorithm"""
        if self.num_qubits < 3:
            return "Need at least 3 qubits for a meaningful Grover demonstration"
        
        n = self.num_qubits
        
        # Reset state
        self.reset_state()
        
        # Apply Hadamard to all qubits to create superposition
        for i in range(n):
            self._hadamard_gate(i)
        
        # Define number of iterations (optimal for small n)
        iterations = int(np.pi/4 * np.sqrt(2**n))
        iterations = max(1, min(iterations, 2))  # Ensure at least 1, at most 2 iterations
        
        # Choose a random target state
        target = np.random.randint(0, 2**n)
        target_binary = format(target, f'0{n}b')
        
        for _ in range(iterations):
            # Oracle - flip sign of target state
            for i in range(2**n):
                if i == target:
                    self.state[i] *= -1
            
            # Diffusion operator (Grover's diffusion)
            # Hadamard all qubits
            for i in range(n):
                self._hadamard_gate(i)
            
            # Inversion about the mean
            # Flip the sign of all states except |0⟩
            for i in range(1, 2**n):
                self.state[i] *= -1
            
            # Hadamard all qubits again
            for i in range(n):
                self._hadamard_gate(i)
        
        # Measure the system
        measurement = self._measure_all()
        
        return f"Grover's algorithm completed after {iterations} iterations.\n" + \
               f"Target state was |{target_binary}⟩.\n{measurement}"
    
    def _run_teleportation(self) -> str:
        """Implement quantum teleportation protocol"""
        if self.num_qubits < 3:
            return "Need at least 3 qubits for quantum teleportation"
        
        # Reset state
        self.reset_state()
        
        # Prepare qubit 0 in a superposition (the state to teleport)
        self._hadamard_gate(0)
        
        # Create Bell pair between qubits 1 and 2
        self._hadamard_gate(1)
        self._cnot_gate(1, 2)
        
        # Begin teleportation
        self._cnot_gate(0, 1)
        self._hadamard_gate(0)
        
        # Measure qubits 0 and 1
        m0 = self._measure_qubit(0)
        m1 = self._measure_qubit(1)
        
        # Apply corrections based on measurement results
        bit0 = "1" in m0  # Extract bit value from the measurement result string
        bit1 = "1" in m1
        
        if bit1:
            self._pauli_x_gate(2)
        if bit0:
            self._pauli_z_gate(2)
        
        return f"Quantum teleportation completed.\nQubit 0 state teleported to qubit 2.\n" + \
               f"Measurements: {m0}, {m1}"
    
    def _run_qft(self) -> str:
        """Implement Quantum Fourier Transform"""
        if self.num_qubits < 2:
            return "Need at least 2 qubits for QFT demonstration"
        
        n = self.num_qubits
        
        # Apply QFT
        for i in range(n):
            self._hadamard_gate(i)
            for j in range(i + 1, n):
                # Phase rotation gates
                phase = np.pi / (2 ** (j - i))
                
                # Implement controlled phase rotation
                for k in range(2**n):
                    if ((k >> i) & 1) and ((k >> j) & 1):
                        self.state[k] *= np.exp(1j * phase)
        
        # Swap qubits (in larger systems)
        if n > 2:
            for i in range(n // 2):
                # Simulate SWAP gates with CNOTs
                self._cnot_gate(i, n - i - 1)
                self._cnot_gate(n - i - 1, i)
                self._cnot_gate(i, n - i - 1)
        
        # Prepare a nice formatted output
        return f"Quantum Fourier Transform applied to {n} qubits"

    def _simulate_rcs(self) -> str:
        """Simulate Random Circuit Sampling (RCS) - similar to Google's quantum supremacy experiment"""
        if self.num_qubits < 5:
            return "Need at least 5 qubits for Random Circuit Sampling simulation"
        
        start_time = time.time()
        
        # Reset to |0⟩ state
        self.reset_state()
        
        # Apply Hadamard to all qubits
        for i in range(self.num_qubits):
            self._hadamard_gate(i)
        
        # Apply random 2-qubit gates in layers
        depth = min(10, 2**self.num_qubits)  # Adjust circuit depth based on qubits
        
        for _ in range(depth):
            # Random single qubit gates
            for i in range(self.num_qubits):
                gate = np.random.choice(["H", "T", "S", "X", "Y", "Z"])
                self.gates[gate](i)
            
            # Random 2-qubit gates
            for i in range(0, self.num_qubits - 1, 2):
                gate = np.random.choice(["CNOT", "CZ"])
                if gate == "CNOT":
                    self._cnot_gate(i, i + 1)
                else:
                    self._cz_gate(i, i + 1)
            
            # Shift pattern for next layer
            for i in range(1, self.num_qubits - 1, 2):
                gate = np.random.choice(["CNOT", "CZ"])
                if gate == "CNOT":
                    self._cnot_gate(i, i + 1)
                else:
                    self._cz_gate(i, i + 1)
        
        # Calculate runtime
        runtime = time.time() - start_time
        
        # Calculate Porter-Thomas distribution statistics to mimic real quantum behavior
        probabilities = np.abs(self.state)**2
        collision_prob = np.sum(probabilities**2)
        
        # Format output
        output = f"Random Circuit Sampling completed in {runtime:.4f} seconds\n"
        output += f"Circuit depth: {depth}\n"
        output += f"Qubits: {self.num_qubits}\n"
        output += f"Hilbert space dimension: {2**self.num_qubits}\n"
        output += f"Collision probability: {collision_prob:.6f}\n\n"
        
        # Simulate taking a few samples
        samples = 5
        output += f"Taking {samples} samples from the quantum state:\n"
        
        for i in range(samples):
            # Sample from the distribution
            sample = np.random.choice(2**self.num_qubits, p=probabilities)
            sample_bin = format(sample, f'0{self.num_qubits}b')
            output += f"Sample {i+1}: |{sample_bin}⟩ with probability {probabilities[sample]:.6f}\n"
        
        return output
    
    def _show_state(self) -> str:
        """Show the current quantum state in readable format"""
        output = "Current quantum state:\n"
        
        # If state is too large, only show non-zero amplitudes
        if self.num_qubits > 5:
            # Find indices with non-zero amplitudes
            indices = np.where(np.abs(self.state) > 1e-10)[0]
            
            if len(indices) == 0:
                output += "State appears to be zero (check for numerical errors)"
                return output
            
            if len(indices) > 20:
                output += f"State has {len(indices)} non-zero components. Showing top 20:\n"
                # Sort by absolute value
                indices = indices[np.argsort(-np.abs(self.state[indices]))]
                indices = indices[:20]
            
            for idx in indices:
                binary = format(idx, f'0{self.num_qubits}b')
                amp = self.state[idx]
                output += f"|{binary}⟩: {amp:.4f}\n"
            
            return output
        
        # For small states, show everything
        total_prob = 0
        for i in range(2**self.num_qubits):
            binary = format(i, f'0{self.num_qubits}b')
            amp = self.state[i]
            prob = np.abs(amp)**2
            total_prob += prob
            
            if np.abs(amp) > 1e-10:  # Only show non-zero amplitudes
                output += f"|{binary}⟩: {amp:.4f} (prob: {prob:.4f})\n"
        
        output += f"\nTotal probability: {total_prob:.10f}"
        return output


def process_quantum_command(command):
    """Global function to process quantum commands for the simulator"""
    if not hasattr(process_quantum_command, "simulator"):
        # Initialize the simulator with 5 qubits on first call
        process_quantum_command.simulator = MCPQuantumSimulator(5)
    
    return process_quantum_command.simulator.process_command(command)


# Example usage:
if __name__ == "__main__":
    # Create a simulator instance
    simulator = MCPQuantumSimulator(5)
    
    # Process some example commands
    print(simulator.process_command("reset system"))
    print(simulator.process_command("apply H to qubit 0"))
    print(simulator.process_command("apply CNOT with control 0 and target 1"))
    print(simulator.process_command("show state"))
    print(simulator.process_command("measure"))
    print(simulator.process_command("run bell algorithm"))
    print(simulator.process_command("simulate random circuit sampling"))
