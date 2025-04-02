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
    def __init__(self, num_qubits=3):
        self.num_qubits = num_qubits
        self.reset_state()
        self.command_patterns = self._define_command_patterns()
        
    def reset_state(self):
        """Reset quantum state to |0⟩^⊗n"""
        self.state = np.zeros(2**self.num_qubits, dtype=complex)
        self.state[0] = 1.0  # Initialize to |0⟩^⊗n
        
    def _define_command_patterns(self):
        """Define regex patterns for natural language commands"""
        return {
            'reset': r'(?i)reset\s+(?:the\s+)?(?:system|simulator|quantum\s+state)',
            'add_qubits': r'(?i)add\s+(\d+)\s+qubits?',
            'apply_gate': r'(?i)apply\s+([HXYZSCT])\s+(?:gate\s+)?(?:to\s+)?(?:qubit\s+)?(\d+)',
            'apply_cnot': r'(?i)apply\s+CNOT\s+(?:gate\s+)?(?:with\s+)?control\s+(\d+)\s+and\s+target\s+(\d+)',
            'apply_cz': r'(?i)apply\s+CZ\s+(?:gate\s+)?(?:with\s+)?control\s+(\d+)\s+and\s+target\s+(\d+)',
            'show_state': r'(?i)show\s+(?:quantum\s+)?state',
            'measure': r'(?i)measure(?:\s+(?:qubit\s+)?(\d+))?',
            'run_algorithm': r'(?i)run\s+(\w+)(?:\s+algorithm)?',
            'simulate_rcs': r'(?i)simulate\s+random\s+circuit\s+sampling(?:\s+with\s+depth\s+(\d+))?'
        }
        
    def process_command(self, command: str) -> str:
        """Process natural language command and return response"""
        for cmd_type, pattern in self.command_patterns.items():
            match = re.match(pattern, command)
            if match:
                if cmd_type == 'reset':
                    self.reset_state()
                    return "System reset to |0⟩^⊗n state"
                
                elif cmd_type == 'add_qubits':
                    num = int(match.group(1))
                    self.num_qubits += num
                    self.reset_state()
                    return f"Added {num} qubits. System now has {self.num_qubits} qubits."
                
                elif cmd_type == 'apply_gate':
                    gate = match.group(1).upper()
                    qubit = int(match.group(2))
                    if qubit >= self.num_qubits:
                        return f"Error: Qubit {qubit} does not exist. System has {self.num_qubits} qubits (0-{self.num_qubits-1})."
                    self._apply_single_qubit_gate(gate, qubit)
                    return f"Applied {gate} gate to qubit {qubit}"
                
                elif cmd_type == 'apply_cnot':
                    control = int(match.group(1))
                    target = int(match.group(2))
                    if max(control, target) >= self.num_qubits:
                        return f"Error: Qubit {max(control, target)} does not exist. System has {self.num_qubits} qubits (0-{self.num_qubits-1})."
                    self._apply_cnot(control, target)
                    return f"Applied CNOT gate with control={control} and target={target}"
                
                elif cmd_type == 'apply_cz':
                    control = int(match.group(1))
                    target = int(match.group(2))
                    if max(control, target) >= self.num_qubits:
                        return f"Error: Qubit {max(control, target)} does not exist. System has {self.num_qubits} qubits (0-{self.num_qubits-1})."
                    self._apply_cz(control, target)
                    return f"Applied CZ gate with control={control} and target={target}"
                
                elif cmd_type == 'show_state':
                    return self._format_quantum_state()
                
                elif cmd_type == 'measure':
                    qubit_str = match.group(1)
                    if qubit_str:
                        qubit = int(qubit_str)
                        if qubit >= self.num_qubits:
                            return f"Error: Qubit {qubit} does not exist. System has {self.num_qubits} qubits (0-{self.num_qubits-1})."
                        result = self._measure_qubit(qubit)
                        return f"Measurement of qubit {qubit} result: |{result}⟩"
                    else:
                        result = self._measure_all()
                        return f"Measurement result: |{result}⟩"
                
                elif cmd_type == 'run_algorithm':
                    algo = match.group(1).lower()
                    return self._run_algorithm(algo)
                
                elif cmd_type == 'simulate_rcs':
                    depth_str = match.group(1)
                    depth = int(depth_str) if depth_str else 5
                    return self._simulate_random_circuit(depth)
        
        return "Command not recognized. Try commands like 'reset system', 'apply H to qubit 0', or 'measure'."
    
    def _apply_single_qubit_gate(self, gate: str, qubit: int):
        """Apply single-qubit gate to specified qubit"""
        # Define standard gates
        if gate == 'H':  # Hadamard
            matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        elif gate == 'X':  # Pauli-X (NOT)
            matrix = np.array([[0, 1], [1, 0]])
        elif gate == 'Y':  # Pauli-Y
            matrix = np.array([[0, -1j], [1j, 0]])
        elif gate == 'Z':  # Pauli-Z
            matrix = np.array([[1, 0], [0, -1]])
        elif gate == 'S':  # Phase gate
            matrix = np.array([[1, 0], [0, 1j]])
        elif gate == 'T':  # T gate
            matrix = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])
        else:
            raise ValueError(f"Unknown gate: {gate}")
        
        # Apply gate to specified qubit
        num_states = 2**self.num_qubits
        new_state = np.zeros(num_states, dtype=complex)
        
        for i in range(num_states):
            # Check if qubit is 0 or 1 in state i
            qubit_val = (i >> qubit) & 1
            
            # Apply gate matrix
            for new_val in [0, 1]:
                # Compute new state index
                new_idx = i & ~(1 << qubit)  # Clear qubit's bit
                new_idx |= (new_val << qubit)  # Set qubit's bit to new value
                
                # Apply gate matrix element
                new_state[new_idx] += self.state[i] * matrix[new_val, qubit_val]
        
        self.state = new_state
    
    def _apply_cnot(self, control: int, target: int):
        """Apply CNOT gate with specified control and target qubits"""
        num_states = 2**self.num_qubits
        new_state = np.zeros(num_states, dtype=complex)
        
        for i in range(num_states):
            # Check if control qubit is 1
            control_val = (i >> control) & 1
            
            if control_val == 0:
                # If control is 0, target remains unchanged
                new_state[i] = self.state[i]
            else:
                # If control is 1, flip target
                target_val = (i >> target) & 1
                new_target = 1 - target_val
                
                # Compute new state index
                new_idx = i & ~(1 << target)  # Clear target's bit
                new_idx |= (new_target << target)  # Set target's bit to new value
                
                new_state[new_idx] = self.state[i]
        
        self.state = new_state
    
    def _apply_cz(self, control: int, target: int):
        """Apply CZ gate with specified control and target qubits"""
        num_states = 2**self.num_qubits
        new_state = np.copy(self.state)
        
        for i in range(num_states):
            # Check if both control and target qubits are 1
            control_val = (i >> control) & 1
            target_val = (i >> target) & 1
            
            if control_val == 1 and target_val == 1:
                # If both qubits are 1, apply phase flip
                new_state[i] *= -1
        
        self.state = new_state
    
    def _format_quantum_state(self) -> str:
        """Format quantum state for display"""
        num_states = 2**self.num_qubits
        result = ["Current quantum state:"]
        
        # Count total probability for verification
        total_prob = 0
        
        # Find states with non-zero amplitude
        for i in range(num_states):
            amplitude = self.state[i]
            if abs(amplitude) > 1e-10:  # Threshold for considering non-zero
                # Convert index to binary representation
                binary = format(i, f'0{self.num_qubits}b')
                probability = abs(amplitude)**2
                total_prob += probability
                
                # Format complex number nicely
                if amplitude.imag == 0:
                    amp_str = f"{amplitude.real:.4f}"
                else:
                    amp_str = f"{amplitude.real:.4f} + {amplitude.imag:.4f}i"
                
                result.append(f"|{binary}⟩: {amp_str} (prob: {probability:.4f})")
        
        result.append(f"\nTotal probability: {total_prob:.10f}")
        return "\n".join(result)
    
    def _measure_qubit(self, qubit: int) -> int:
        """Measure specified qubit and collapse the state"""
        num_states = 2**self.num_qubits
        prob_zero = 0
        
        # Calculate probability of measuring 0
        for i in range(num_states):
            if ((i >> qubit) & 1) == 0:  # If qubit is 0 in this state
                prob_zero += abs(self.state[i])**2
        
        # Randomly choose outcome based on probabilities
        if np.random.random() < prob_zero:
            outcome = 0
        else:
            outcome = 1
        
        # Collapse state based on measurement
        new_state = np.zeros(num_states, dtype=complex)
        norm_factor = 0
        
        for i in range(num_states):
            if ((i >> qubit) & 1) == outcome:  # If qubit matches outcome
                new_state[i] = self.state[i]
                norm_factor += abs(self.state[i])**2
        
        # Normalize the new state
        if norm_factor > 0:
            new_state /= np.sqrt(norm_factor)
        
        self.state = new_state
        return outcome
    
    def _measure_all(self) -> str:
        """Measure all qubits and return result as binary string"""
        # Calculate probabilities for all states
        probs = np.abs(self.state)**2
        
        # Choose one state based on probabilities
        result_idx = np.random.choice(2**self.num_qubits, p=probs)
        
        # Convert to binary string
        result = format(result_idx, f'0{self.num_qubits}b')
        
        # Collapse state to measured result
        new_state = np.zeros_like(self.state)
        new_state[result_idx] = 1.0
        self.state = new_state
        
        return result
    
    def _run_algorithm(self, algorithm: str) -> str:
        """Run specified quantum algorithm"""
        if algorithm == 'bell':
            return self._create_bell_state()
        elif algorithm == 'deutsch' or algorithm == 'deutschjozsa':
            return self._run_deutsch_jozsa()
        elif algorithm == 'grover':
            return self._run_grover()
        elif algorithm == 'teleportation':
            return self._run_teleportation()
        elif algorithm == 'qft':
            return self._run_qft()
        else:
            return f"Unknown algorithm: {algorithm}. Available algorithms: bell, deutsch, grover, teleportation, qft"
    
    def _create_bell_state(self) -> str:
        """Create a Bell state between qubits 0 and 1"""
        if self.num_qubits < 2:
            return "Error: Bell state requires at least 2 qubits"
        
        self.reset_state()
        self._apply_single_qubit_gate('H', 0)
        self._apply_cnot(0, 1)
        
        return "Created Bell state between qubits 0 and 1"
    
    def _run_deutsch_jozsa(self) -> str:
        """Run Deutsch-Jozsa algorithm"""
        if self.num_qubits < 2:
            return "Error: Deutsch-Jozsa algorithm requires at least 2 qubits"
        
        # Implement a simple version with balanced oracle
        self.reset_state()
        
        # Apply H to all qubits
        for i in range(self.num_qubits):
            self._apply_single_qubit_gate('H', i)
        
        # Apply oracle (use CNOT as a balanced function)
        self._apply_cnot(0, self.num_qubits-1)
        
        # Apply H to non-ancilla qubits
        for i in range(self.num_qubits-1):
            self._apply_single_qubit_gate('H', i)
        
        # Measure
        result = self._measure_all()
        
        if result.startswith('0' * (self.num_qubits-1)):
            return "Deutsch-Jozsa result: Constant function"
        else:
            return "Deutsch-Jozsa result: Balanced function"
    
    def _run_grover(self) -> str:
        """Run Grover's search algorithm"""
        if self.num_qubits < 3:
            return "Error: Grover's algorithm demonstration requires at least 3 qubits"
        
        n = self.num_qubits
        self.reset_state()
        
        # Prepare superposition
        for i in range(n):
            self._apply_single_qubit_gate('H', i)
        
        # Define target state (for demonstration, use |101...1⟩)
        target = (1 << n) - 1 - (1 << (n-2))  # |101...1⟩
        
        # Number of iterations (optimal for small n)
        iterations = int(np.floor(np.pi/4 * np.sqrt(2**n)))
        
        # Run Grover iterations
        for _ in range(1):  # Simplified for demonstration
            # Oracle operation (mark target state)
            for i in range(2**n):
                if i == target:
                    self.state[i] *= -1
            
            # Diffusion operator
            # H on all qubits
            for i in range(n):
                self._apply_single_qubit_gate('H', i)
            
            # Phase inversion about mean
            for i in range(2**n):
                if i == 0:  # |00...0⟩
                    self.state[i] *= -1
            
            # H on all qubits
            for i in range(n):
                self._apply_single_qubit_gate('H', i)
        
        # Measure
        result = self._measure_all()
        
        return f"Grover's search result: |{result}⟩ (target was |{format(target, f'0{n}b')}⟩)"
    
    def _run_teleportation(self) -> str:
        """Run quantum teleportation protocol"""
        if self.num_qubits < 3:
            return "Error: Quantum teleportation requires at least 3 qubits"
        
        self.reset_state()
        
        # Prepare state to teleport (qubit 0)
        self._apply_single_qubit_gate('H', 0)
        self._apply_single_qubit_gate('T', 0)
        
        # Create Bell pair between qubits 1 and 2
        self._apply_single_qubit_gate('H', 1)
        self._apply_cnot(1, 2)
        
        # Bell measurement between qubits 0 and 1
        self._apply_cnot(0, 1)
        self._apply_single_qubit_gate('H', 0)
        
        # Measure qubits 0 and 1
        m0 = self._measure_qubit(0)
        m1 = self._measure_qubit(1)
        
        # Apply corrections to qubit 2
        if m1 == 1:
            self._apply_single_qubit_gate('X', 2)
        if m0 == 1:
            self._apply_single_qubit_gate('Z', 2)
        
        return f"Teleported quantum state from qubit 0 to qubit 2. Measurements: m0={m0}, m1={m1}"
    
    def _run_qft(self) -> str:
        """Run Quantum Fourier Transform"""
        if self.num_qubits < 2:
            return "Error: QFT demonstration requires at least 2 qubits"
        
        n = self.num_qubits
        self.reset_state()
        
        # Prepare a simple state
        self._apply_single_qubit_gate('X', 0)
        
        # Apply QFT
        for i in range(n):
            # Apply H to qubit i
            self._apply_single_qubit_gate('H', i)
            
            # Apply controlled rotations
            for j in range(i+1, n):
                # Simulate controlled phase rotation
                # In a full implementation, would need controlled-R_k gates
                self._apply_cz(i, j)
        
        # In a full implementation, would need qubit swaps here
        
        return f"Applied Quantum Fourier Transform to {n} qubits"
    
    def _simulate_random_circuit(self, depth: int = 5) -> str:
        """Simulate a random quantum circuit (inspired by quantum supremacy experiments)"""
        if self.num_qubits < 3:
            return "Error: Random Circuit Sampling requires at least 3 qubits"
        
        n = self.num_qubits
        self.reset_state()
        
        # Apply initial layer of Hadamards
        for i in range(n):
            self._apply_single_qubit_gate('H', i)
        
        # Apply random gates in layers
        gates = ['H', 'X', 'Y', 'Z', 'S', 'T']
        
        for d in range(depth):
            # Apply random single-qubit gates
            for i in range(n):
                gate = np.random.choice(gates)
                self._apply_single_qubit_gate(gate, i)
            
            # Apply random two-qubit gates
            for i in range(n-1):
                if np.random.random() < 0.5:
                    self._apply_cnot(i, (i+1) % n)
                else:
                    self._apply_cz(i, (i+1) % n)
        
        # Sample
        samples = []
        original_state = np.copy(self.state)
        
        for _ in range(5):  # Take 5 samples
            self.state = np.copy(original_state)
            result = self._measure_all()
            samples.append(result)
        
        self.state = original_state
        
        return f"Random Circuit Sampling (depth={depth}) results:\n" + "\n".join([f"|{s}⟩" for s in samples])

# Create a persistent simulator instance
_simulator = MCPQuantumSimulator(3)

def process_quantum_command(command: str) -> str:
    """Process a quantum command using the persistent simulator"""
    return _simulator.process_command(command)

# Interactive mode when run directly
if __name__ == "__main__":
    print("🌌 MCP Quantum Computer Simulator 🚀")
    print("Enter commands in natural language (or 'exit' to quit):")
    print("Examples:")
    print("  - reset system")
    print("  - add 2 qubits")
    print("  - apply H to qubit 0")
    print("  - apply CNOT with control 0 and target 1")
    print("  - show state")
    print("  - measure")
    print("  - run bell algorithm")
    print("  - simulate random circuit sampling")
    
    while True:
        cmd = input("\n> ")
        if cmd.lower() in ('exit', 'quit', 'q'):
            break
        
        result = process_quantum_command(cmd)
        print(result)