# راهنمای کامل اجزای فرانت‌اند CognifyX

## 📋 خلاصه

پروژه CognifyX شامل دو فرانت‌اند اصلی است که هر کدام ویژگی‌های منحصر به فردی دارند:

1. **Consciousness UI** - رابط کاربری هوشیاری پیشرفته
2. **Ultimate Frontend** - نسخه نهایی با قابلیت‌های پیشرفته

---

## 🧠 Consciousness UI - رابط کاربری هوشیاری

### 🏗️ معماری کلی
```jsx
// ساختار اصلی App.jsx
├── Hooks پیشرفته
│   ├── useConsciousnessEngine() - مدیریت حالت هوشیاری
│   ├── useNeuralQuantumState() - مدیریت حالت عصبی-کوانتومی
│   └── useCognitiveMetrics() - مدیریت معیارهای شناختی
├── کامپوننت‌های اصلی
│   ├── HolographicMetricCard - کارت‌های متریک هولوگرافیک
│   ├── NeuromorphicSection - بخش‌های نورومورفیک
│   ├── QuantumProgressBar - نوارهای پیشرفت کوانتومی
│   └── ConsciousnessDashboard - داشبورد هوشیاری
└── انیمیشن‌ها و افکت‌ها
    ├── Framer Motion - انیمیشن‌های پیشرفته
    └── CSS Gradients - افکت‌های بصری
```

### 🔧 Hooks پیشرفته

#### 1. useConsciousnessEngine
```jsx
const useConsciousnessEngine = () => {
  // مدیریت حالت هوشیاری
  const [consciousnessState, setConsciousnessState] = useState({
    level: 'AWAKENING',        // سطح هوشیاری
    depth: 0.76,              // عمق
    clarity: 0.83,            // وضوح
    awareness: 0.71,          // آگاهی
    metacognition: 0.68,      // فراشناخت
    transcendence: 0.12,      // تعالی
    unity: 0.08,              // وحدت
    presence: 0.89,           // حضور
    flow: {                   // حالت جریان
      state: 0.84,
      challenge_skill_balance: 0.79,
      clear_goals: 0.91,
      immediate_feedback: 0.87,
      action_awareness_merge: 0.73,
      time_transformation: 0.65,
      autotelic_experience: 0.71
    }
  });
}
```

#### 2. useNeuralQuantumState
```jsx
const useNeuralQuantumState = () => {
  // مدیریت حالت عصبی-کوانتومی
  const [neuralQuantumState, setNeuralQuantumState] = useState({
    eeg: {                    // الکتروانسفالوگرافی
      channels: 128,
      sampling_rate: 2000,
      signal_quality: 0.94,
      artifacts: 3,
      coherence: 0.87,
      power_spectrum: {       // طیف قدرت امواج مغزی
        delta: { power: 0.23, coherence: 0.78, phase: 1.2 },
        theta: { power: 0.31, coherence: 0.82, phase: 2.1 },
        alpha: { power: 0.45, coherence: 0.91, phase: 0.8 },
        beta: { power: 0.67, coherence: 0.76, phase: 1.9 },
        gamma: { power: 0.52, coherence: 0.83, phase: 2.7 }
      },
      connectivity: {         // اتصالات عصبی
        frontal_parietal: 0.89,
        default_mode: 0.34,
        salience: 0.76,
        executive: 0.82
      }
    },
    fnirs: {                  // طیف‌سنجی نزدیک مادون قرمز
      channels: 52,
      oxy_hb: 0.73,          // هموگلوبین اکسیژنه
      deoxy_hb: 0.27,        // هموگلوبین دی‌اکسیژنه
      total_hb: 1.12,        // کل هموگلوبین
      signal_to_noise: 23.4,
      hemodynamic_response: 0.68,
      oxygenation_index: 0.81
    },
    quantum: {                // حالت کوانتومی
      qubits: 256,
      coherence_time: 847,
      entanglement_fidelity: 0.987,
      gate_errors: 0.0003,
      quantum_advantage: 12.7,
      operations_completed: 47291,
      superposition_states: 128,
      decoherence_rate: 0.0012
    }
  });
}
```

#### 3. useCognitiveMetrics
```jsx
const useCognitiveMetrics = () => {
  // مدیریت معیارهای شناختی
  const [cognitiveMetrics, setCognitiveMetrics] = useState({
    attention: {              // توجه
      sustained: 0.84,       // پایدار
      selective: 0.79,       // انتخابی
      divided: 0.62,         // تقسیم شده
      executive: 0.88,       // اجرایی
      vigilance: 0.71,       // هوشیاری
      focus_duration: 847    // مدت تمرکز
    },
    memory: {                 // حافظه
      working: 0.76,         // کاری
      episodic: 0.68,        // رویدادی
      semantic: 0.82,        // معنایی
      procedural: 0.91,      // رویه‌ای
      capacity: 0.73,        // ظرفیت
      retrieval_speed: 0.85  // سرعت بازیابی
    },
    processing: {             // پردازش
      speed: 0.89,           // سرعت
      accuracy: 0.94,        // دقت
      flexibility: 0.71,     // انعطاف‌پذیری
      inhibition: 0.78,      // بازداری
      switching: 0.83,       // تغییر
      updating: 0.76         // به‌روزرسانی
    },
    emotional: {              // عاطفی
      valence: 0.23,         // ظرفیت
      arousal: 0.67,         // برانگیختگی
      regulation: 0.84,      // تنظیم
      empathy: 0.79,         // همدلی
      social_cognition: 0.72, // شناخت اجتماعی
      emotional_intelligence: 0.86 // هوش عاطفی
    },
    creativity: {             // خلاقیت
      divergent_thinking: 0.78, // تفکر واگرا
      convergent_thinking: 0.82, // تفکر همگرا
      originality: 0.69,     // اصالت
      fluency: 0.85,         // روانی
      flexibility: 0.73,     // انعطاف‌پذیری
      elaboration: 0.77      // بسط
    }
  });
}
```

### 🎨 کامپوننت‌های اصلی

#### 1. HolographicMetricCard
```jsx
const HolographicMetricCard = ({ 
  title, 
  value, 
  unit = '', 
  icon: Icon, 
  gradientFrom, 
  gradientTo,
  glowIntensity = 'medium' 
}) => {
  return (
    <motion.div
      className={`relative overflow-hidden rounded-xl p-6 bg-gradient-to-br ${gradientFrom} ${gradientTo} 
                 border border-white/20 backdrop-blur-sm shadow-2xl`}
      whileHover={{ scale: 1.02, y: -2 }}
      transition={{ duration: 0.2 }}
    >
      {/* محتوای کارت */}
    </motion.div>
  );
};
```

#### 2. NeuromorphicSection
```jsx
const NeuromorphicSection = ({ 
  title, 
  icon: Icon, 
  iconColor, 
  children 
}) => {
  return (
    <div className="relative p-6 rounded-2xl bg-gradient-to-br from-gray-900/50 to-gray-800/30 
                    border border-white/10 backdrop-blur-sm shadow-xl">
      {/* هدر بخش */}
      <div className="flex items-center gap-3 mb-6">
        <Icon className={`w-6 h-6 ${iconColor}`} />
        <h2 className="text-xl font-semibold text-white">{title}</h2>
      </div>
      {/* محتوای بخش */}
      {children}
    </div>
  );
};
```

#### 3. QuantumProgressBar
```jsx
const QuantumProgressBar = ({ 
  value, 
  label, 
  color, 
  glowIntensity = 'medium' 
}) => {
  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm">
        <span className="text-gray-300">{label}</span>
        <span className="text-white font-medium">
          {formatPercentage(value)}
        </span>
      </div>
      <div className="relative h-3 bg-gray-800 rounded-full overflow-hidden">
        <motion.div
          className={`h-full ${color} rounded-full`}
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 1, ease: "easeOut" }}
        />
      </div>
    </div>
  );
};
```

### 🎭 انیمیشن‌ها و افکت‌ها

#### 1. Framer Motion
```jsx
// انیمیشن‌های ورود
<motion.div
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
  transition={{ duration: 0.6, ease: "easeOut" }}
>

// انیمیشن‌های hover
<motion.div
  whileHover={{ scale: 1.05, rotate: 2 }}
  whileTap={{ scale: 0.95 }}
  transition={{ duration: 0.2 }}
>

// انیمیشن‌های تغییر view
<AnimatePresence mode="wait">
  {activeView === 'consciousness' && (
    <motion.div
      key="consciousness"
      initial={{ opacity: 0, x: 100 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -100 }}
      transition={{ duration: 0.5 }}
    >
```

#### 2. CSS Gradients و افکت‌ها
```css
/* گرادیان‌های پیشرفته */
.bg-gradient-to-br {
  background: linear-gradient(135deg, var(--from-color), var(--to-color));
}

/* افکت‌های شیشه‌ای */
.backdrop-blur-sm {
  backdrop-filter: blur(8px);
}

/* سایه‌های نئون */
.shadow-neon {
  box-shadow: 0 0 20px rgba(59, 130, 246, 0.5);
}
```

---

## ⭐ Ultimate Frontend - نسخه نهایی

### 🏗️ معماری کلی
```jsx
// ساختار اصلی App.jsx
├── Hooks پیشرفته
│   ├── useConsciousnessEngine() - موتور هوشیاری پیشرفته
│   ├── useSessionManager() - مدیریت جلسات
│   └── useDataVisualization() - تجسم داده‌ها
├── کامپوننت‌های UI
│   ├── Card, CardContent, CardHeader - کارت‌های اطلاعات
│   ├── Progress - نوارهای پیشرفت
│   ├── Badge - نشان‌های وضعیت
│   ├── Tabs, TabsContent - تب‌های اطلاعات
│   ├── Slider - اسلایدرهای تنظیم
│   └── Switch - کلیدهای تغییر وضعیت
└── ویژگی‌های پیشرفته
    ├── Real-time Data Streaming - جریان داده‌های زنده
    ├── Advanced Analytics - تحلیل‌های پیشرفته
    └── Interactive Controls - کنترل‌های تعاملی
```

### 🔧 Hooks پیشرفته

#### 1. useConsciousnessEngine (نسخه پیشرفته)
```jsx
const useConsciousnessEngine = () => {
  const [sessionActive, setSessionActive] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [realTimeData, setRealTimeData] = useState({
    consciousness: {
      level: 'AWAKENING',
      depth: 0.76,
      clarity: 0.83,
      stability: 0.71,
      awareness: {
        self: 0.68,           // آگاهی از خود
        environmental: 0.74,  // آگاهی محیطی
        temporal: 0.82,       // آگاهی زمانی
        metacognitive: 0.65   // آگاهی فراشناختی
      },
      transcendence: {
        unity: 0.12,          // وحدت
        transcendence: 0.08,  // تعالی
        ego_dissolution: 0.05 // انحلال خود
      },
      flow: {
        intensity: 0.89,      // شدت جریان
        challenge_skill_balance: 0.92,
        action_awareness_merge: 0.87,
        clear_goals: 0.94,
        immediate_feedback: 0.88,
        time_transformation: 0.76,
        autotelic_experience: 0.91
      }
    },
    neural: {
      eeg: {
        delta: 0.097,         // امواج دلتا
        theta: 0.069,         // امواج تتا
        alpha: 0.266,         // امواج آلفا
        beta: 0.575,          // امواج بتا
        gamma: 0.532          // امواج گاما
      },
      fnirs: {
        oxygenated_hb: 0.738,     // هموگلوبین اکسیژنه
        deoxygenated_hb: 0.276,   // هموگلوبین دی‌اکسیژنه
        oxygenation_index: 0.810  // شاخص اکسیژناسیون
      },
      signal_quality: 0.94,
      artifacts: 2
    },
    quantum: {
      active_qubits: 256,         // کیوبیت‌های فعال
      coherence_time: 813.34,     // زمان انسجام
      entanglement_fidelity: 0.98, // وفاداری درهم‌تنیدگی
      gate_error_rate: 0.002,     // نرخ خطای گیت
      quantum_advantage: 11.91,   // مزیت کوانتومی
      operations_completed: 1847293, // عملیات تکمیل شده
      processing_efficiency: 0.967 // کارایی پردازش
    },
    cognitive: {
      attention: {
        sustained: 0.765,     // توجه پایدار
        selective: 0.732,     // توجه انتخابی
        executive: 0.873      // توجه اجرایی
      },
      memory: {
        working: 0.747,       // حافظه کاری
        episodic: 0.593,      // حافظه رویدادی
        semantic: 0.820       // حافظه معنایی
      },
      processing: {
        speed: 0.834,         // سرعت پردازش
        accuracy: 0.912,      // دقت پردازش
        flexibility: 0.698    // انعطاف‌پذیری
      },
      creativity: {
        divergent_thinking: 0.788, // تفکر واگرا
        originality: 0.690,        // اصالت
        fluency: 0.850             // روانی
      },
      emotional: {
        awareness: 0.823,     // آگاهی عاطفی
        regulation: 0.756,    // تنظیم عاطفی
        empathy: 0.789        // همدلی
      }
    }
  });
}
```

### 🎨 کامپوننت‌های UI

#### 1. Card Components
```jsx
// کارت‌های اطلاعات
<Card className="bg-gradient-to-br from-gray-900/50 to-gray-800/30 border-white/10">
  <CardHeader>
    <CardTitle className="text-white">Consciousness Level</CardTitle>
    <CardDescription className="text-gray-400">
      Current state of consciousness
    </CardDescription>
  </CardHeader>
  <CardContent>
    <div className="space-y-4">
      <Progress value={consciousnessData.depth * 100} />
      <div className="flex justify-between text-sm">
        <span className="text-gray-300">Depth</span>
        <span className="text-white">{formatPercentage(consciousnessData.depth)}</span>
      </div>
    </div>
  </CardContent>
</Card>
```

#### 2. Tabs Components
```jsx
// تب‌های اطلاعات
<Tabs defaultValue="consciousness" className="w-full">
  <TabsList className="grid w-full grid-cols-4">
    <TabsTrigger value="consciousness">Consciousness</TabsTrigger>
    <TabsTrigger value="neural">Neural</TabsTrigger>
    <TabsTrigger value="quantum">Quantum</TabsTrigger>
    <TabsTrigger value="cognitive">Cognitive</TabsTrigger>
  </TabsList>
  
  <TabsContent value="consciousness">
    <ConsciousnessView data={consciousnessData} />
  </TabsContent>
  
  <TabsContent value="neural">
    <NeuralView data={neuralData} />
  </TabsContent>
  
  <TabsContent value="quantum">
    <QuantumView data={quantumData} />
  </TabsContent>
  
  <TabsContent value="cognitive">
    <CognitiveView data={cognitiveData} />
  </TabsContent>
</Tabs>
```

#### 3. Interactive Controls
```jsx
// اسلایدرهای تنظیم
<Slider
  value={[consciousnessSettings.depth]}
  onValueChange={(value) => updateConsciousnessSettings('depth', value[0])}
  max={1}
  step={0.01}
  className="w-full"
/>

// کلیدهای تغییر وضعیت
<Switch
  checked={sessionActive}
  onCheckedChange={setSessionActive}
  className="data-[state=checked]:bg-blue-600"
/>
```

---

## 🎯 تفاوت‌های کلیدی

### Consciousness UI
- **تمرکز**: رابط کاربری هوشیاری پیشرفته
- **ویژگی‌ها**: 
  - انیمیشن‌های هولوگرافیک
  - افکت‌های بصری پیشرفته
  - کامپوننت‌های سفارشی
  - تجربه کاربری غوطه‌ور

### Ultimate Frontend
- **تمرکز**: رابط کاربری جامع و کاربردی
- **ویژگی‌ها**:
  - کامپوننت‌های استاندارد UI
  - کنترل‌های تعاملی
  - تجسم داده‌های پیشرفته
  - مدیریت جلسات

---

## 🚀 نحوه استفاده

### نصب وابستگی‌ها
```bash
# برای Consciousness UI
cd cognifyx-consciousness-ui
pnpm install

# برای Ultimate Frontend
cd cognifyx-ultimate-frontend
pnpm install
```

### اجرای پروژه
```bash
# اجرای توسعه
pnpm dev

# ساخت برای تولید
pnpm build

# پیش‌نمایش تولید
pnpm preview
```

### تکنولوژی‌های استفاده شده
- **React 18** - کتابخانه اصلی
- **Vite** - ابزار ساخت
- **Framer Motion** - انیمیشن‌ها
- **Lucide React** - آیکون‌ها
- **Tailwind CSS** - استایل‌دهی
- **Radix UI** - کامپوننت‌های پایه

---

## 📊 معیارهای عملکرد

### Consciousness UI
- **اندازه bundle**: ~2.5MB
- **زمان بارگذاری**: <3 ثانیه
- **انیمیشن‌ها**: 60fps
- **تعامل**: <100ms

### Ultimate Frontend
- **اندازه bundle**: ~1.8MB
- **زمان بارگذاری**: <2 ثانیه
- **پاسخ‌دهی**: <50ms
- **دسترسی‌پذیری**: WCAG 2.1 AA

---

## 🔧 توسعه و سفارشی‌سازی

### اضافه کردن کامپوننت جدید
```jsx
// مثال: کامپوننت جدید برای نمایش داده‌ها
const CustomDataCard = ({ title, value, icon: Icon }) => {
  return (
    <Card className="bg-gradient-to-br from-purple-900/50 to-purple-800/30">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Icon className="w-5 h-5 text-purple-400" />
          <CardTitle className="text-white">{title}</CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold text-purple-300">
          {value}
        </div>
      </CardContent>
    </Card>
  );
};
```

### سفارشی‌سازی تم
```css
/* متغیرهای CSS سفارشی */
:root {
  --consciousness-primary: #3b82f6;
  --consciousness-secondary: #8b5cf6;
  --consciousness-accent: #06b6d4;
  --consciousness-background: #0f172a;
  --consciousness-surface: #1e293b;
}
```

---

## 📝 نتیجه‌گیری

هر دو فرانت‌اند CognifyX ویژگی‌های منحصر به فردی دارند:

- **Consciousness UI**: برای تجربه‌های غوطه‌ور و نمایش‌های پیشرفته
- **Ultimate Frontend**: برای کاربردهای عملی و مدیریت داده‌ها

انتخاب بین این دو بستگی به نیازهای خاص پروژه دارد.
