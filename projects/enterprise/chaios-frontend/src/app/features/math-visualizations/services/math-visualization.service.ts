import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, interval, map } from 'rxjs';

export interface MathEquation {
  id: string;
  name: string;
  category: 'consciousness' | 'quantum' | 'fractal' | 'chaos' | 'topology' | 'number_theory';
  latex: string;
  description: string;
  parameters: { [key: string]: MathParameter };
  complexity: 'basic' | 'intermediate' | 'advanced' | 'research';
  visualization: VisualizationType;
  interactive: boolean;
  realTime: boolean;
}

export interface MathParameter {
  name: string;
  symbol: string;
  type: 'number' | 'range' | 'complex' | 'vector' | 'matrix';
  value: any;
  min?: number;
  max?: number;
  step?: number;
  description: string;
}

export interface VisualizationConfig {
  equation: string;
  parameters: { [key: string]: any };
  renderMode: '2d' | '3d' | 'complex' | 'phase' | 'parametric';
  colorScheme: 'default' | 'consciousness' | 'quantum' | 'rainbow' | 'monochrome';
  animation: boolean;
  gridLines: boolean;
  axes: boolean;
  labels: boolean;
  resolution: number;
  bounds: {
    xMin: number;
    xMax: number;
    yMin: number;
    yMax: number;
    zMin?: number;
    zMax?: number;
  };
}

export interface VisualizationData {
  points: Point3D[];
  curves: Curve[];
  surfaces: Surface[];
  vectors: Vector3D[];
  annotations: Annotation[];
  metadata: {
    equation: string;
    timestamp: Date;
    computationTime: number;
    complexity: number;
  };
}

export interface Point3D {
  x: number;
  y: number;
  z: number;
  color?: string;
  size?: number;
  label?: string;
}

export interface Curve {
  points: Point3D[];
  color: string;
  width: number;
  style: 'solid' | 'dashed' | 'dotted';
}

export interface Surface {
  vertices: Point3D[];
  faces: number[][];
  color: string;
  opacity: number;
  wireframe: boolean;
}

export interface Vector3D {
  origin: Point3D;
  direction: Point3D;
  magnitude: number;
  color: string;
  label?: string;
}

export interface Annotation {
  position: Point3D;
  text: string;
  color: string;
  size: number;
}

export type VisualizationType = 
  | 'function_2d' 
  | 'function_3d' 
  | 'parametric_2d' 
  | 'parametric_3d' 
  | 'complex_plane' 
  | 'phase_portrait' 
  | 'vector_field' 
  | 'fractal' 
  | 'differential_equation'
  | 'consciousness_field';

@Injectable({
  providedIn: 'root'
})
export class MathVisualizationService {
  private currentEquationSubject = new BehaviorSubject<MathEquation | null>(null);
  private visualizationDataSubject = new BehaviorSubject<VisualizationData | null>(null);
  private isComputingSubject = new BehaviorSubject<boolean>(false);
  private animationFrameSubject = new BehaviorSubject<number>(0);

  public currentEquation$ = this.currentEquationSubject.asObservable();
  public visualizationData$ = this.visualizationDataSubject.asObservable();
  public isComputing$ = this.isComputingSubject.asObservable();
  public animationFrame$ = this.animationFrameSubject.asObservable();

  // Real-time animation stream
  public animationStream$ = interval(16).pipe( // ~60 FPS
    map(frame => frame * 0.016) // Convert to seconds
  );

  private predefinedEquations: MathEquation[] = [
    // Consciousness Equations
    {
      id: 'wallace-transform',
      name: 'Wallace Transform',
      category: 'consciousness',
      latex: '\\Psi(x,t) = \\sum_{n=1}^{\\infty} \\frac{\\phi^n}{\\sqrt{n}} e^{i(kx - \\omega t + \\phi_n)}',
      description: 'Consciousness field equation using golden ratio harmonics',
      parameters: {
        phi: { name: 'Golden Ratio', symbol: 'φ', type: 'number', value: 1.618033988749, description: 'The golden ratio constant' },
        omega: { name: 'Frequency', symbol: 'ω', type: 'range', value: 1.0, min: 0.1, max: 10.0, step: 0.1, description: 'Angular frequency' },
        k: { name: 'Wave Number', symbol: 'k', type: 'range', value: 1.0, min: 0.1, max: 5.0, step: 0.1, description: 'Wave number' },
        harmonics: { name: 'Harmonics', symbol: 'n', type: 'range', value: 10, min: 1, max: 50, step: 1, description: 'Number of harmonics' }
      },
      complexity: 'advanced',
      visualization: 'consciousness_field',
      interactive: true,
      realTime: true
    },
    {
      id: 'mobius-optimization',
      name: 'Möbius Optimization',
      category: 'consciousness',
      latex: 'f(z) = \\frac{az + b}{cz + d}, \\quad ad - bc \\neq 0',
      description: 'Möbius transformation for consciousness optimization',
      parameters: {
        a: { name: 'Parameter A', symbol: 'a', type: 'complex', value: { real: 1, imag: 0 }, description: 'Complex parameter a' },
        b: { name: 'Parameter B', symbol: 'b', type: 'complex', value: { real: 0, imag: 1 }, description: 'Complex parameter b' },
        c: { name: 'Parameter C', symbol: 'c', type: 'complex', value: { real: 1, imag: 0 }, description: 'Complex parameter c' },
        d: { name: 'Parameter D', symbol: 'd', type: 'complex', value: { real: 0, imag: 1 }, description: 'Complex parameter d' }
      },
      complexity: 'advanced',
      visualization: 'complex_plane',
      interactive: true,
      realTime: false
    },

    // Quantum Equations
    {
      id: 'schrodinger',
      name: 'Schrödinger Equation',
      category: 'quantum',
      latex: 'i\\hbar\\frac{\\partial}{\\partial t}|\\psi\\rangle = \\hat{H}|\\psi\\rangle',
      description: 'Time-dependent Schrödinger equation',
      parameters: {
        hbar: { name: 'Reduced Planck Constant', symbol: 'ℏ', type: 'number', value: 1.054571817e-34, description: 'Reduced Planck constant' },
        mass: { name: 'Particle Mass', symbol: 'm', type: 'range', value: 1.0, min: 0.1, max: 10.0, step: 0.1, description: 'Particle mass' },
        potential: { name: 'Potential Energy', symbol: 'V', type: 'range', value: 0.0, min: -10.0, max: 10.0, step: 0.1, description: 'Potential energy' }
      },
      complexity: 'advanced',
      visualization: 'function_3d',
      interactive: true,
      realTime: true
    },

    // Fractal Equations
    {
      id: 'mandelbrot',
      name: 'Mandelbrot Set',
      category: 'fractal',
      latex: 'z_{n+1} = z_n^2 + c',
      description: 'The famous Mandelbrot fractal set',
      parameters: {
        maxIterations: { name: 'Max Iterations', symbol: 'n', type: 'range', value: 100, min: 10, max: 1000, step: 10, description: 'Maximum iterations' },
        escapeRadius: { name: 'Escape Radius', symbol: 'r', type: 'range', value: 2.0, min: 1.0, max: 10.0, step: 0.1, description: 'Escape radius' },
        zoom: { name: 'Zoom Level', symbol: 'z', type: 'range', value: 1.0, min: 0.1, max: 1000.0, step: 0.1, description: 'Zoom level' }
      },
      complexity: 'intermediate',
      visualization: 'fractal',
      interactive: true,
      realTime: false
    },
    {
      id: 'julia',
      name: 'Julia Set',
      category: 'fractal',
      latex: 'z_{n+1} = z_n^2 + c',
      description: 'Julia set fractal with complex parameter c',
      parameters: {
        c_real: { name: 'C Real Part', symbol: 'Re(c)', type: 'range', value: -0.7, min: -2.0, max: 2.0, step: 0.01, description: 'Real part of c' },
        c_imag: { name: 'C Imaginary Part', symbol: 'Im(c)', type: 'range', value: 0.27015, min: -2.0, max: 2.0, step: 0.01, description: 'Imaginary part of c' },
        maxIterations: { name: 'Max Iterations', symbol: 'n', type: 'range', value: 100, min: 10, max: 1000, step: 10, description: 'Maximum iterations' }
      },
      complexity: 'intermediate',
      visualization: 'fractal',
      interactive: true,
      realTime: true
    },

    // Chaos Theory
    {
      id: 'lorenz-attractor',
      name: 'Lorenz Attractor',
      category: 'chaos',
      latex: '\\frac{dx}{dt} = \\sigma(y - x), \\frac{dy}{dt} = x(\\rho - z) - y, \\frac{dz}{dt} = xy - \\beta z',
      description: 'The famous chaotic Lorenz system',
      parameters: {
        sigma: { name: 'Sigma', symbol: 'σ', type: 'range', value: 10.0, min: 1.0, max: 50.0, step: 0.1, description: 'Prandtl number' },
        rho: { name: 'Rho', symbol: 'ρ', type: 'range', value: 28.0, min: 1.0, max: 100.0, step: 0.1, description: 'Rayleigh number' },
        beta: { name: 'Beta', symbol: 'β', type: 'range', value: 8/3, min: 0.1, max: 10.0, step: 0.1, description: 'Geometric factor' }
      },
      complexity: 'advanced',
      visualization: 'parametric_3d',
      interactive: true,
      realTime: true
    },

    // Number Theory
    {
      id: 'riemann-zeta',
      name: 'Riemann Zeta Function',
      category: 'number_theory',
      latex: '\\zeta(s) = \\sum_{n=1}^{\\infty} \\frac{1}{n^s} = \\prod_p \\frac{1}{1-p^{-s}}',
      description: 'The Riemann zeta function and its zeros',
      parameters: {
        s_real: { name: 'S Real Part', symbol: 'Re(s)', type: 'range', value: 0.5, min: -10.0, max: 10.0, step: 0.1, description: 'Real part of s' },
        s_imag: { name: 'S Imaginary Part', symbol: 'Im(s)', type: 'range', value: 14.134725, min: 0.0, max: 100.0, step: 0.1, description: 'Imaginary part of s' },
        terms: { name: 'Series Terms', symbol: 'n', type: 'range', value: 1000, min: 10, max: 10000, step: 10, description: 'Number of series terms' }
      },
      complexity: 'research',
      visualization: 'complex_plane',
      interactive: true,
      realTime: false
    },

    // Topology
    {
      id: 'klein-bottle',
      name: 'Klein Bottle',
      category: 'topology',
      latex: 'x = (a + b\\cos(v))\\cos(u), y = (a + b\\cos(v))\\sin(u), z = b\\sin(v)',
      description: 'Klein bottle - a non-orientable surface',
      parameters: {
        a: { name: 'Major Radius', symbol: 'a', type: 'range', value: 3.0, min: 1.0, max: 10.0, step: 0.1, description: 'Major radius' },
        b: { name: 'Minor Radius', symbol: 'b', type: 'range', value: 1.0, min: 0.1, max: 5.0, step: 0.1, description: 'Minor radius' },
        u_range: { name: 'U Parameter Range', symbol: 'u', type: 'range', value: 6.28, min: 3.14, max: 12.56, step: 0.1, description: 'U parameter range' },
        v_range: { name: 'V Parameter Range', symbol: 'v', type: 'range', value: 6.28, min: 3.14, max: 12.56, step: 0.1, description: 'V parameter range' }
      },
      complexity: 'advanced',
      visualization: 'parametric_3d',
      interactive: true,
      realTime: false
    }
  ];

  constructor() {
    this.loadPredefinedEquations();
  }

  private loadPredefinedEquations(): void {
    // Initialize with Wallace Transform as default
    this.currentEquationSubject.next(this.predefinedEquations[0]);
  }

  getEquationsByCategory(category?: string): MathEquation[] {
    if (!category) {
      return this.predefinedEquations;
    }
    return this.predefinedEquations.filter(eq => eq.category === category);
  }

  getEquationById(id: string): MathEquation | null {
    return this.predefinedEquations.find(eq => eq.id === id) || null;
  }

  setCurrentEquation(equation: MathEquation): void {
    this.currentEquationSubject.next(equation);
    this.computeVisualization(equation);
  }

  updateParameter(parameterName: string, value: any): void {
    const currentEquation = this.currentEquationSubject.value;
    if (!currentEquation) return;

    const updatedEquation = {
      ...currentEquation,
      parameters: {
        ...currentEquation.parameters,
        [parameterName]: {
          ...currentEquation.parameters[parameterName],
          value: value
        }
      }
    };

    this.currentEquationSubject.next(updatedEquation);
    
    if (updatedEquation.realTime) {
      this.computeVisualization(updatedEquation);
    }
  }

  async computeVisualization(equation: MathEquation, config?: Partial<VisualizationConfig>): Promise<void> {
    this.isComputingSubject.next(true);
    
    try {
      const startTime = performance.now();
      
      const visualizationConfig: VisualizationConfig = {
        equation: equation.latex,
        parameters: this.extractParameterValues(equation.parameters),
        renderMode: this.getRenderModeForVisualization(equation.visualization),
        colorScheme: 'consciousness',
        animation: equation.realTime,
        gridLines: true,
        axes: true,
        labels: true,
        resolution: 100,
        bounds: {
          xMin: -10,
          xMax: 10,
          yMin: -10,
          yMax: 10,
          zMin: -10,
          zMax: 10
        },
        ...config
      };

      const data = await this.generateVisualizationData(equation, visualizationConfig);
      
      const computationTime = performance.now() - startTime;
      data.metadata.computationTime = computationTime;
      data.metadata.timestamp = new Date();
      
      this.visualizationDataSubject.next(data);
      
    } catch (error) {
      console.error('Visualization computation failed:', error);
    } finally {
      this.isComputingSubject.next(false);
    }
  }

  private extractParameterValues(parameters: { [key: string]: MathParameter }): { [key: string]: any } {
    const values: { [key: string]: any } = {};
    
    Object.entries(parameters).forEach(([key, param]) => {
      values[key] = param.value;
    });
    
    return values;
  }

  private getRenderModeForVisualization(visualization: VisualizationType): '2d' | '3d' | 'complex' | 'phase' | 'parametric' {
    switch (visualization) {
      case 'function_2d':
      case 'parametric_2d':
        return '2d';
      case 'function_3d':
      case 'parametric_3d':
      case 'consciousness_field':
        return '3d';
      case 'complex_plane':
        return 'complex';
      case 'phase_portrait':
        return 'phase';
      default:
        return 'parametric';
    }
  }

  private async generateVisualizationData(equation: MathEquation, config: VisualizationConfig): Promise<VisualizationData> {
    const data: VisualizationData = {
      points: [],
      curves: [],
      surfaces: [],
      vectors: [],
      annotations: [],
      metadata: {
        equation: equation.latex,
        timestamp: new Date(),
        computationTime: 0,
        complexity: this.calculateComplexity(equation, config)
      }
    };

    switch (equation.visualization) {
      case 'consciousness_field':
        await this.generateConsciousnessField(equation, config, data);
        break;
      case 'function_3d':
        await this.generateFunction3D(equation, config, data);
        break;
      case 'parametric_3d':
        await this.generateParametric3D(equation, config, data);
        break;
      case 'complex_plane':
        await this.generateComplexPlane(equation, config, data);
        break;
      case 'fractal':
        await this.generateFractal(equation, config, data);
        break;
      case 'function_2d':
        await this.generateFunction2D(equation, config, data);
        break;
      case 'parametric_2d':
        await this.generateParametric2D(equation, config, data);
        break;
      case 'phase_portrait':
        await this.generatePhasePortrait(equation, config, data);
        break;
      case 'vector_field':
        await this.generateVectorField(equation, config, data);
        break;
      case 'differential_equation':
        await this.generateDifferentialEquation(equation, config, data);
        break;
      default:
        await this.generateFunction2D(equation, config, data);
    }

    return data;
  }

  private async generateConsciousnessField(equation: MathEquation, config: VisualizationConfig, data: VisualizationData): Promise<void> {
    const params = config.parameters;
    const phi = params.phi || 1.618033988749;
    const omega = params.omega || 1.0;
    const k = params.k || 1.0;
    const harmonics = params.harmonics || 10;
    
    const resolution = config.resolution;
    const time = this.animationFrameSubject.value * 0.1;
    
    // Generate consciousness field points
    for (let i = 0; i <= resolution; i++) {
      for (let j = 0; j <= resolution; j++) {
        const x = config.bounds.xMin + (i / resolution) * (config.bounds.xMax - config.bounds.xMin);
        const y = config.bounds.yMin + (j / resolution) * (config.bounds.yMax - config.bounds.yMin);
        
        let z = 0;
        let phase = 0;
        
        // Wallace Transform computation
        for (let n = 1; n <= harmonics; n++) {
          const amplitude = Math.pow(phi, n) / Math.sqrt(n);
          const phaseShift = Math.PI * n / harmonics;
          const contribution = amplitude * Math.cos(k * x - omega * time + phaseShift);
          
          z += contribution;
          phase += phaseShift;
        }
        
        // Color based on phase and amplitude
        const hue = (phase * 180 / Math.PI) % 360;
        const intensity = Math.abs(z) / 10;
        const color = `hsl(${hue}, 70%, ${50 + intensity * 50}%)`;
        
        data.points.push({
          x, y, z,
          color,
          size: Math.max(1, intensity * 5)
        });
      }
    }

    // Add golden ratio spiral annotation
    data.annotations.push({
      position: { x: 0, y: 0, z: 5 },
      text: `φ = ${phi.toFixed(6)}`,
      color: '#FFD700',
      size: 14
    });
  }

  private async generateFunction3D(equation: MathEquation, config: VisualizationConfig, data: VisualizationData): Promise<void> {
    const resolution = config.resolution;
    const vertices: Point3D[] = [];
    const faces: number[][] = [];
    
    // Generate grid of points
    for (let i = 0; i <= resolution; i++) {
      for (let j = 0; j <= resolution; j++) {
        const x = config.bounds.xMin + (i / resolution) * (config.bounds.xMax - config.bounds.xMin);
        const y = config.bounds.yMin + (j / resolution) * (config.bounds.yMax - config.bounds.yMin);
        
        // Evaluate function at (x, y)
        const z = this.evaluateFunction(equation, { x, y, ...config.parameters });
        
        vertices.push({ x, y, z });
        
        // Create faces for surface mesh
        if (i > 0 && j > 0) {
          const current = i * (resolution + 1) + j;
          const left = current - 1;
          const below = (i - 1) * (resolution + 1) + j;
          const belowLeft = below - 1;
          
          faces.push([belowLeft, left, current]);
          faces.push([belowLeft, current, below]);
        }
      }
    }
    
    data.surfaces.push({
      vertices,
      faces,
      color: '#3880ff',
      opacity: 0.8,
      wireframe: false
    });
  }

  private async generateParametric3D(equation: MathEquation, config: VisualizationConfig, data: VisualizationData): Promise<void> {
    const resolution = config.resolution;
    const points: Point3D[] = [];
    
    // Generate parametric curve/surface
    for (let t = 0; t <= 1; t += 1 / resolution) {
      const params = { t, ...config.parameters };
      
      if (equation.id === 'lorenz-attractor') {
        // Special handling for Lorenz attractor
        const trajectory = this.computeLorenzTrajectory(params, 1000);
        trajectory.forEach(point => data.points.push(point));
        
        // Create curve from trajectory
        data.curves.push({
          points: trajectory,
          color: '#ff6b6b',
          width: 2,
          style: 'solid'
        });
      } else {
        const point = this.evaluateParametricFunction(equation, params);
        points.push(point);
      }
    }
    
    if (points.length > 0 && equation.id !== 'lorenz-attractor') {
      data.curves.push({
        points,
        color: '#4ecdc4',
        width: 3,
        style: 'solid'
      });
    }
  }

  private async generateComplexPlane(equation: MathEquation, config: VisualizationConfig, data: VisualizationData): Promise<void> {
    const resolution = config.resolution;
    
    for (let i = 0; i <= resolution; i++) {
      for (let j = 0; j <= resolution; j++) {
        const real = config.bounds.xMin + (i / resolution) * (config.bounds.xMax - config.bounds.xMin);
        const imag = config.bounds.yMin + (j / resolution) * (config.bounds.yMax - config.bounds.yMin);
        
        const z = { real, imag };
        const result = this.evaluateComplexFunction(equation, z, config.parameters);
        
        // Map complex result to color
        const magnitude = Math.sqrt(result.real * result.real + result.imag * result.imag);
        const phase = Math.atan2(result.imag, result.real);
        
        const hue = (phase * 180 / Math.PI + 360) % 360;
        const saturation = Math.min(magnitude * 50, 100);
        const lightness = 50;
        
        data.points.push({
          x: real,
          y: imag,
          z: magnitude,
          color: `hsl(${hue}, ${saturation}%, ${lightness}%)`,
          size: 2
        });
      }
    }
  }

  private async generateFractal(equation: MathEquation, config: VisualizationConfig, data: VisualizationData): Promise<void> {
    const resolution = config.resolution * 2; // Higher resolution for fractals
    const maxIterations = config.parameters.maxIterations || 100;
    
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = config.bounds.xMin + (i / resolution) * (config.bounds.xMax - config.bounds.xMin);
        const y = config.bounds.yMin + (j / resolution) * (config.bounds.yMax - config.bounds.yMin);
        
        let iterations: number;
        
        if (equation.id === 'mandelbrot') {
          iterations = this.mandelbrotIterations(x, y, maxIterations);
        } else if (equation.id === 'julia') {
          const c = {
            real: config.parameters.c_real || -0.7,
            imag: config.parameters.c_imag || 0.27015
          };
          iterations = this.juliaIterations(x, y, c, maxIterations);
        } else {
          iterations = maxIterations;
        }
        
        // Color based on iterations
        const hue = (iterations / maxIterations) * 360;
        const saturation = iterations === maxIterations ? 0 : 100;
        const lightness = iterations === maxIterations ? 0 : 50 + (iterations / maxIterations) * 50;
        
        data.points.push({
          x, y, z: 0,
          color: `hsl(${hue}, ${saturation}%, ${lightness}%)`,
          size: 1
        });
      }
    }
  }

  private async generateFunction2D(equation: MathEquation, config: VisualizationConfig, data: VisualizationData): Promise<void> {
    const resolution = config.resolution;
    const points: Point3D[] = [];
    
    for (let i = 0; i <= resolution; i++) {
      const x = config.bounds.xMin + (i / resolution) * (config.bounds.xMax - config.bounds.xMin);
      const y = this.evaluateFunction(equation, { x, ...config.parameters });
      
      points.push({ x, y, z: 0 });
    }
    
    data.curves.push({
      points,
      color: '#3880ff',
      width: 2,
      style: 'solid'
    });
  }

  private async generateParametric2D(equation: MathEquation, config: VisualizationConfig, data: VisualizationData): Promise<void> {
    const resolution = config.resolution;
    const points: Point3D[] = [];
    
    for (let i = 0; i <= resolution; i++) {
      const t = i / resolution;
      const point = this.evaluateParametricFunction(equation, { t, ...config.parameters });
      points.push({ ...point, z: 0 });
    }
    
    data.curves.push({
      points,
      color: '#e74c3c',
      width: 2,
      style: 'solid'
    });
  }

  private async generatePhasePortrait(equation: MathEquation, config: VisualizationConfig, data: VisualizationData): Promise<void> {
    // Phase portrait implementation
    const resolution = config.resolution / 2;
    
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = config.bounds.xMin + (i / resolution) * (config.bounds.xMax - config.bounds.xMin);
        const y = config.bounds.yMin + (j / resolution) * (config.bounds.yMax - config.bounds.yMin);
        
        // Compute vector field
        const dx = this.evaluateDerivative(equation, { x, y, ...config.parameters }, 'x');
        const dy = this.evaluateDerivative(equation, { x, y, ...config.parameters }, 'y');
        
        const magnitude = Math.sqrt(dx * dx + dy * dy);
        const normalizedDx = dx / magnitude;
        const normalizedDy = dy / magnitude;
        
        data.vectors.push({
          origin: { x, y, z: 0 },
          direction: { x: normalizedDx, y: normalizedDy, z: 0 },
          magnitude: Math.min(magnitude, 2),
          color: `hsl(${(Math.atan2(dy, dx) * 180 / Math.PI + 360) % 360}, 70%, 50%)`
        });
      }
    }
  }

  private async generateVectorField(equation: MathEquation, config: VisualizationConfig, data: VisualizationData): Promise<void> {
    // Similar to phase portrait but for 3D vector fields
    await this.generatePhasePortrait(equation, config, data);
  }

  private async generateDifferentialEquation(equation: MathEquation, config: VisualizationConfig, data: VisualizationData): Promise<void> {
    // Numerical solution of differential equations
    const resolution = config.resolution;
    const dt = 0.01;
    const points: Point3D[] = [];
    
    let x = config.bounds.xMin;
    let y = 0; // Initial condition
    
    for (let i = 0; i <= resolution; i++) {
      points.push({ x, y, z: 0 });
      
      // Simple Euler method
      const dy = this.evaluateDerivative(equation, { x, y, ...config.parameters }, 'x');
      y += dy * dt;
      x += dt;
    }
    
    data.curves.push({
      points,
      color: '#9b59b6',
      width: 2,
      style: 'solid'
    });
  }

  // Helper methods for mathematical computations
  private evaluateFunction(equation: MathEquation, variables: any): number {
    // Simplified function evaluation - in practice, this would use a math parser
    switch (equation.id) {
      case 'schrodinger':
        return Math.sin(variables.x) * Math.exp(-variables.x * variables.x / 10);
      default:
        return Math.sin(variables.x);
    }
  }

  private evaluateParametricFunction(equation: MathEquation, params: any): Point3D {
    const t = params.t;
    
    switch (equation.id) {
      case 'klein-bottle':
        const a = params.a || 3;
        const b = params.b || 1;
        const u = t * 2 * Math.PI;
        const v = t * 2 * Math.PI;
        
        return {
          x: (a + b * Math.cos(v)) * Math.cos(u),
          y: (a + b * Math.cos(v)) * Math.sin(u),
          z: b * Math.sin(v)
        };
      default:
        return { x: t, y: Math.sin(t * 2 * Math.PI), z: 0 };
    }
  }

  private evaluateComplexFunction(equation: MathEquation, z: { real: number; imag: number }, params: any): { real: number; imag: number } {
    switch (equation.id) {
      case 'mobius-optimization':
        const a = params.a || { real: 1, imag: 0 };
        const b = params.b || { real: 0, imag: 1 };
        const c = params.c || { real: 1, imag: 0 };
        const d = params.d || { real: 0, imag: 1 };
        
        // (az + b) / (cz + d)
        const numerator = this.complexMultiply(a, z);
        const denominator = this.complexAdd(this.complexMultiply(c, z), d);
        
        return this.complexDivide(this.complexAdd(numerator, b), denominator);
      default:
        return z;
    }
  }

  private complexAdd(a: { real: number; imag: number }, b: { real: number; imag: number }): { real: number; imag: number } {
    return { real: a.real + b.real, imag: a.imag + b.imag };
  }

  private complexMultiply(a: { real: number; imag: number }, b: { real: number; imag: number }): { real: number; imag: number } {
    return {
      real: a.real * b.real - a.imag * b.imag,
      imag: a.real * b.imag + a.imag * b.real
    };
  }

  private complexDivide(a: { real: number; imag: number }, b: { real: number; imag: number }): { real: number; imag: number } {
    const denominator = b.real * b.real + b.imag * b.imag;
    return {
      real: (a.real * b.real + a.imag * b.imag) / denominator,
      imag: (a.imag * b.real - a.real * b.imag) / denominator
    };
  }

  private mandelbrotIterations(x: number, y: number, maxIterations: number): number {
    let zx = 0, zy = 0;
    let iteration = 0;
    
    while (zx * zx + zy * zy < 4 && iteration < maxIterations) {
      const temp = zx * zx - zy * zy + x;
      zy = 2 * zx * zy + y;
      zx = temp;
      iteration++;
    }
    
    return iteration;
  }

  private juliaIterations(x: number, y: number, c: { real: number; imag: number }, maxIterations: number): number {
    let zx = x, zy = y;
    let iteration = 0;
    
    while (zx * zx + zy * zy < 4 && iteration < maxIterations) {
      const temp = zx * zx - zy * zy + c.real;
      zy = 2 * zx * zy + c.imag;
      zx = temp;
      iteration++;
    }
    
    return iteration;
  }

  private computeLorenzTrajectory(params: any, steps: number): Point3D[] {
    const sigma = params.sigma || 10;
    const rho = params.rho || 28;
    const beta = params.beta || 8/3;
    const dt = 0.01;
    
    let x = 1, y = 1, z = 1;
    const points: Point3D[] = [];
    
    for (let i = 0; i < steps; i++) {
      // Lorenz equations
      const dx = sigma * (y - x);
      const dy = x * (rho - z) - y;
      const dz = x * y - beta * z;
      
      x += dx * dt;
      y += dy * dt;
      z += dz * dt;
      
      const hue = (i / steps) * 360;
      points.push({
        x, y, z,
        color: `hsl(${hue}, 70%, 50%)`,
        size: 1
      });
    }
    
    return points;
  }

  private evaluateDerivative(equation: MathEquation, variables: any, variable: string): number {
    // Numerical derivative approximation
    const h = 0.001;
    const vars1 = { ...variables };
    const vars2 = { ...variables };
    
    vars1[variable] -= h;
    vars2[variable] += h;
    
    const y1 = this.evaluateFunction(equation, vars1);
    const y2 = this.evaluateFunction(equation, vars2);
    
    return (y2 - y1) / (2 * h);
  }

  private calculateComplexity(equation: MathEquation, config: VisualizationConfig): number {
    let complexity = 1;
    
    // Base complexity from equation type
    switch (equation.complexity) {
      case 'basic': complexity *= 1; break;
      case 'intermediate': complexity *= 2; break;
      case 'advanced': complexity *= 4; break;
      case 'research': complexity *= 8; break;
    }
    
    // Resolution impact
    complexity *= (config.resolution / 100) ** 2;
    
    // Visualization type impact
    switch (equation.visualization) {
      case 'function_2d': complexity *= 1; break;
      case 'parametric_2d': complexity *= 1.5; break;
      case 'function_3d': complexity *= 3; break;
      case 'parametric_3d': complexity *= 4; break;
      case 'complex_plane': complexity *= 2; break;
      case 'fractal': complexity *= 5; break;
      case 'consciousness_field': complexity *= 6; break;
      default: complexity *= 2;
    }
    
    return complexity;
  }

  startAnimation(): void {
    // Animation is handled by the animationStream$ observable
    interval(16).subscribe(frame => {
      this.animationFrameSubject.next(frame);
      
      const currentEquation = this.currentEquationSubject.value;
      if (currentEquation?.realTime) {
        this.computeVisualization(currentEquation);
      }
    });
  }

  exportVisualization(format: 'json' | 'csv' | 'svg' | 'png'): any {
    const data = this.visualizationDataSubject.value;
    const equation = this.currentEquationSubject.value;
    
    if (!data || !equation) return null;
    
    switch (format) {
      case 'json':
        return {
          equation: equation,
          visualization: data,
          timestamp: new Date().toISOString()
        };
      case 'csv':
        return this.convertToCSV(data);
      default:
        return data;
    }
  }

  private convertToCSV(data: VisualizationData): string {
    const headers = ['x', 'y', 'z', 'color', 'size', 'type'];
    const rows: string[] = [headers.join(',')];
    
    data.points.forEach(point => {
      rows.push(`${point.x},${point.y},${point.z},${point.color || ''},${point.size || 1},point`);
    });
    
    return rows.join('\n');
  }
}
