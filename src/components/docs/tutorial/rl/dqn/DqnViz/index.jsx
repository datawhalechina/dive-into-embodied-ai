import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Pause, RotateCcw, SkipForward, Brain, Database, Activity } from 'lucide-react';

// --- 常量定义 ---
const GRID_SIZE = 5;
const CELL_SIZE = 60;
const ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT'];
const ACTION_VECTORS = {
  UP: { x: 0, y: -1 },
  DOWN: { x: 0, y: 1 },
  LEFT: { x: -1, y: 0 },
  RIGHT: { x: 1, y: 0 },
};

const TILE_TYPES = {
  EMPTY: 0,
  WALL: 1,
  TRAP: 2,
  GOAL: 3,
};

// 预定义地图 (0:空, 1:墙, 2:陷阱, 3:终点)
const INITIAL_GRID = [
  [0, 0, 0, 0, 0],
  [0, 1, 1, 0, 0],
  [0, 0, 2, 0, 0],
  [0, 1, 2, 1, 0],
  [0, 0, 0, 0, 3],
];

// 简单的 Q-Table 模拟神经网络行为
class SimulatedDQN {
  constructor() {
    this.qTable = {};
    this.learningRate = 0.2;
    this.discountFactor = 0.9;
  }

  getQ(x, y) {
    const key = `${x},${y}`;
    if (!this.qTable[key]) {
      this.qTable[key] = { UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0 };
    }
    return this.qTable[key];
  }

  predict(x, y) {
    return this.getQ(x, y);
  }

  train(state, action, reward, nextState, done) {
    const currentQ = this.getQ(state.x, state.y)[action];
    const nextQValues = this.getQ(nextState.x, nextState.y);
    const maxNextQ = done ? 0 : Math.max(...Object.values(nextQValues));
    const targetQ = reward + this.discountFactor * maxNextQ;
    const loss = Math.pow(targetQ - currentQ, 2);
    this.qTable[`${state.x},${state.y}`][action] += this.learningRate * (targetQ - currentQ);
    return { loss, targetQ, currentQ };
  }

  reset() {
    this.qTable = {};
  }
}

const dqnAgent = new SimulatedDQN();

const DqnVisualization = () => {
  const [agentPos, setAgentPos] = useState({ x: 0, y: 0 });
  const [isTraining, setIsTraining] = useState(false);
  const [epsilon, setEpsilon] = useState(0.5);
  const [episode, setEpisode] = useState(0);
  const [stepCount, setStepCount] = useState(0);
  const [totalReward, setTotalReward] = useState(0);
  const [memory, setMemory] = useState([]);
  const [lastAction, setLastAction] = useState(null);
  const [lossHistory, setLossHistory] = useState([]);
  const [speed, setSpeed] = useState(500);
  const [gridRefresh, setGridRefresh] = useState(0);
  const [mode, setMode] = useState('TRAIN');

  const timerRef = useRef(null);

  const resetGame = useCallback(() => {
    setAgentPos({ x: 0, y: 0 });
    setStepCount(0);
    setTotalReward(0);
    setLastAction(null);
  }, []);

  const fullReset = () => {
    setIsTraining(false);
    dqnAgent.reset();
    setEpsilon(0.5);
    setEpisode(0);
    setMemory([]);
    setLossHistory([]);
    resetGame();
    setGridRefresh(prev => prev + 1);
  };

  const step = useCallback(() => {
    const state = { ...agentPos };
    let action = '';

    if (Math.random() < epsilon && mode === 'TRAIN') {
      action = ACTIONS[Math.floor(Math.random() * ACTIONS.length)];
    } else {
      const qValues = dqnAgent.predict(state.x, state.y);
      const bestActions = Object.keys(qValues).filter(k => qValues[k] === Math.max(...Object.values(qValues)));
      action = bestActions[Math.floor(Math.random() * bestActions.length)];
    }

    const vector = ACTION_VECTORS[action];
    let nextX = state.x + vector.x;
    let nextY = state.y + vector.y;
    let reward = -0.1;
    let done = false;

    if (nextX < 0 || nextX >= GRID_SIZE || nextY < 0 || nextY >= GRID_SIZE) {
      nextX = state.x;
      nextY = state.y;
      reward = -0.5;
    } else if (INITIAL_GRID[nextY][nextX] === TILE_TYPES.WALL) {
      nextX = state.x;
      nextY = state.y;
      reward = -0.5;
    }

    const tileType = INITIAL_GRID[nextY][nextX];
    if (tileType === TILE_TYPES.TRAP) {
      reward = -5;
      done = true;
    } else if (tileType === TILE_TYPES.GOAL) {
      reward = 10;
      done = true;
    }

    const nextState = { x: nextX, y: nextY };

    if (mode === 'TRAIN') {
      const experience = { state, action, reward, nextState, done, id: Date.now() + Math.random() };
      setMemory(prev => [experience, ...prev].slice(0, 20));
      const { loss } = dqnAgent.train(state, action, reward, nextState, done);
      if (memory.length > 5) {
        const randomExp = memory[Math.floor(Math.random() * memory.length)];
        dqnAgent.train(randomExp.state, randomExp.action, randomExp.reward, randomExp.nextState, randomExp.done);
      }
      setLossHistory(prev => [...prev, loss].slice(-50));
    }

    setAgentPos(nextState);
    setTotalReward(prev => prev + reward);
    setStepCount(prev => prev + 1);
    setLastAction({ action, isRandom: Math.random() < epsilon && mode === 'TRAIN' });
    setGridRefresh(prev => prev + 1);

    if (done) {
      setEpisode(prev => prev + 1);
      if (mode === 'TRAIN') {
        setEpsilon(prev => Math.max(0.01, prev * 0.98));
      }
      resetGame();
    }
  }, [agentPos, epsilon, memory, mode, resetGame]);

  useEffect(() => {
    if (isTraining) {
      timerRef.current = setInterval(step, speed);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [isTraining, step, speed]);

  // 神经网络可视化
  const NetworkViz = () => {
    const qValues = dqnAgent.predict(agentPos.x, agentPos.y);
    const maxQ = Math.max(...Object.values(qValues));
    const minQ = Math.min(...Object.values(qValues));
    const getBarHeight = (val) => {
      const range = maxQ - minQ;
      if (range === 0) return 50;
      return 20 + ((val - minQ) / range) * 80;
    };

    return (
      <div style={{background:'#0f172a',padding:'16px',borderRadius:'12px',border:'1px solid #334155',display:'flex',flexDirection:'column',alignItems:'center',position:'relative',overflow:'hidden'}}>
        <div style={{position:'absolute',top:8,left:8,fontSize:12,color:'#94a3b8',fontFamily:'monospace'}}>DQN (Deep Q-Network)</div>
        {/* 输入层 */}
        <div style={{display:'flex',gap:16,marginBottom:32,marginTop:24}}>
          {[{label:'X',val:agentPos.x},{label:'Y',val:agentPos.y}].map(n => (
            <div key={n.label} style={{display:'flex',flexDirection:'column',alignItems:'center'}}>
              <div style={{width:40,height:40,borderRadius:'50%',background:'#3b82f6',display:'flex',alignItems:'center',justifyContent:'center',color:'#fff',fontWeight:'bold',boxShadow:'0 0 0 4px rgba(59,130,246,0.2)'}}>
                {n.label}
              </div>
              <span style={{fontSize:12,marginTop:4,color:'#cbd5e1'}}>{n.val}</span>
            </div>
          ))}
        </div>
        {/* 隐藏层 */}
        <div style={{display:'flex',gap:8,marginBottom:32,background:'#1e293b',padding:8,borderRadius:8,border:'1px solid #475569'}}>
          {[1,2,3,4,5].map(i => (
            <div key={i} style={{width:12,height:12,borderRadius:'50%',background:'#c084fc',opacity:0.8}}></div>
          ))}
          <span style={{fontSize:12,color:'#94a3b8',marginLeft:8,alignSelf:'center'}}>Hidden Layers</span>
        </div>
        {/* 输出层 Q-Values */}
        <div style={{display:'flex',gap:12,alignItems:'flex-end',height:96,width:'100%',justifyContent:'center',padding:'0 16px',borderBottom:'1px solid #334155',paddingBottom:8}}>
          {ACTIONS.map(act => {
            const val = qValues[act];
            const isMax = val === maxQ && val !== 0;
            const height = getBarHeight(val);
            return (
              <div key={act} style={{display:'flex',flexDirection:'column',alignItems:'center',width:48}}>
                <div style={{fontSize:10,color:'#94a3b8',marginBottom:4}}>{val.toFixed(2)}</div>
                <div style={{
                  width:'100%',borderRadius:'4px 4px 0 0',
                  transition:'all 0.3s',
                  background: isMax ? '#22c55e' : '#475569',
                  boxShadow: isMax ? '0 0 10px rgba(34,197,94,0.6)' : 'none',
                  height: `${height}%`
                }}></div>
                <div style={{fontSize:10,marginTop:4,fontWeight:'bold',color: isMax ? '#4ade80' : '#64748b'}}>{act}</div>
              </div>
            );
          })}
        </div>
        <div style={{fontSize:12,color:'#64748b',marginTop:8}}>输出: 各动作的Q值 (Q-Values)</div>
      </div>
    );
  };

  // 网格渲染
  const renderGrid = () => {
    const getCellBg = (cellType, maxQ) => {
      if (cellType === TILE_TYPES.WALL) return '#0f172a';
      if (cellType === TILE_TYPES.TRAP) return 'rgba(127,29,29,0.5)';
      if (cellType === TILE_TYPES.GOAL) return 'rgba(20,83,45,0.5)';
      if (maxQ > 0) return `rgba(79,70,229,${Math.min(0.9, maxQ * 0.1 + 0.2)})`;
      return '#334155';
    };

    const getCellBorder = (cellType) => {
      if (cellType === TILE_TYPES.WALL) return '1px solid #475569';
      if (cellType === TILE_TYPES.TRAP) return '1px solid #991b1b';
      if (cellType === TILE_TYPES.GOAL) return '1px solid #166534';
      return 'none';
    };

    return (
      <div style={{
        display:'grid',gap:4,background:'#1e293b',padding:4,borderRadius:8,
        gridTemplateColumns:`repeat(${GRID_SIZE}, ${CELL_SIZE}px)`
      }}>
        {INITIAL_GRID.map((row, y) =>
          row.map((cellType, x) => {
            const isAgent = agentPos.x === x && agentPos.y === y;
            const qVals = dqnAgent.getQ(x, y);
            const maxQ = Math.max(...Object.values(qVals));
            const bestAction = Object.keys(qVals).reduce((a, b) => qVals[a] > qVals[b] ? a : b);
            let arrowRot = 0;
            if (bestAction === 'DOWN') arrowRot = 180;
            if (bestAction === 'LEFT') arrowRot = -90;
            if (bestAction === 'RIGHT') arrowRot = 90;

            return (
              <div key={`${x}-${y}`} style={{
                width: CELL_SIZE, height: CELL_SIZE,
                background: getCellBg(cellType, maxQ),
                border: getCellBorder(cellType),
                borderRadius: 4,
                display:'flex', alignItems:'center', justifyContent:'center',
                position:'relative', transition:'background 0.2s'
              }}>
                {cellType === TILE_TYPES.TRAP && <span style={{fontSize:24}}>🔥</span>}
                {cellType === TILE_TYPES.GOAL && <span style={{fontSize:24}}>🚩</span>}
                {cellType === 0 && Math.abs(maxQ) > 0.01 && (
                  <div style={{position:'absolute',opacity:0.3,color:'#fff',fontWeight:'bold',fontSize:18,transform:`rotate(${arrowRot}deg)`,pointerEvents:'none'}}>
                    ↑
                  </div>
                )}
                {isAgent && (
                  <div style={{position:'absolute',zIndex:10,fontSize:28}}>
                    🤖
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    );
  };

  const statBoxStyle = {padding:8,background:'#1e293b',borderRadius:4};
  const labelStyle = {color:'#94a3b8',fontSize:12};
  const valueStyle = {fontFamily:'monospace',fontSize:20,color:'#fff'};

  return (
    <div style={{background:'#020617',color:'#e2e8f0',padding:16,fontFamily:'sans-serif',display:'flex',flexDirection:'column',alignItems:'center',borderRadius:12,marginTop:16,marginBottom:16}}>
      {/* Header */}
      <header style={{width:'100%',maxWidth:1100,marginBottom:24,display:'flex',justifyContent:'space-between',alignItems:'center',borderBottom:'1px solid #1e293b',paddingBottom:16,flexWrap:'wrap',gap:16}}>
        <div>
          <h2 style={{fontSize:22,fontWeight:'bold',background:'linear-gradient(to right,#60a5fa,#a855f7)',WebkitBackgroundClip:'text',WebkitTextFillColor:'transparent',margin:0}}>
            DQN 算法原理可视化
          </h2>
          <p style={{color:'#94a3b8',fontSize:14,marginTop:4}}>
            Deep Q-Network: 结合深度学习(感知)与强化学习(决策)
          </p>
        </div>
        <div style={{display:'flex',gap:16,fontSize:14,color:'#94a3b8'}}>
          <div style={{display:'flex',alignItems:'center',gap:8}}>
            <span style={{width:12,height:12,background:'rgba(79,70,229,0.5)',borderRadius:3,display:'block'}}></span> 高价值区域
          </div>
          <div style={{display:'flex',alignItems:'center',gap:8}}>
            <span style={{width:12,height:12,background:'rgba(127,29,29,0.5)',borderRadius:3,display:'block',border:'1px solid #991b1b'}}></span> 陷阱
          </div>
          <div style={{display:'flex',alignItems:'center',gap:8}}>
            <span style={{width:12,height:12,background:'rgba(20,83,45,0.5)',borderRadius:3,display:'block',border:'1px solid #166534'}}></span> 目标
          </div>
        </div>
      </header>

      {/* Main Layout */}
      <main style={{width:'100%',maxWidth:1100,display:'grid',gridTemplateColumns:'1fr 1fr 1fr',gap:24}}>
        {/* Left: Game & Controls */}
        <div style={{display:'flex',flexDirection:'column',gap:16}}>
          <div style={{display:'flex',justifyContent:'center',padding:16,background:'#0f172a',borderRadius:12,border:'1px solid #1e293b'}}>
            {renderGrid()}
          </div>
          {/* Stats */}
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:8,background:'#0f172a',padding:12,borderRadius:8,border:'1px solid #1e293b',fontSize:14}}>
            <div style={statBoxStyle}>
              <div style={labelStyle}>Episode</div>
              <div style={valueStyle}>{episode}</div>
            </div>
            <div style={statBoxStyle}>
              <div style={labelStyle}>Total Reward</div>
              <div style={{...valueStyle,color:totalReward > 0 ? '#4ade80' : '#f87171'}}>{totalReward.toFixed(1)}</div>
            </div>
            <div style={statBoxStyle}>
              <div style={labelStyle}>Epsilon (探索率)</div>
              <div style={{...valueStyle,color:'#93c5fd'}}>{(epsilon * 100).toFixed(0)}%</div>
            </div>
            <div style={statBoxStyle}>
              <div style={labelStyle}>Step Count</div>
              <div style={valueStyle}>{stepCount}</div>
            </div>
          </div>
          {/* Controls */}
          <div style={{background:'#0f172a',padding:16,borderRadius:12,border:'1px solid #1e293b',display:'flex',flexDirection:'column',gap:12}}>
            <div style={{display:'flex',gap:8,justifyContent:'center',marginBottom:8}}>
              <button onClick={() => setIsTraining(!isTraining)}
                style={{display:'flex',alignItems:'center',gap:8,padding:'8px 24px',borderRadius:8,fontWeight:'bold',border:'none',cursor:'pointer',
                  background: isTraining ? '#f59e0b' : '#16a34a', color: isTraining ? '#000' : '#fff'}}>
                {isTraining ? <><Pause size={18}/> 暂停</> : <><Play size={18}/> {mode === 'TRAIN' ? '训练' : '运行'}</>}
              </button>
              <button onClick={step} disabled={isTraining}
                style={{padding:8,background:'#334155',borderRadius:8,color:'#fff',border:'none',cursor:'pointer',opacity:isTraining?0.5:1}} title="单步">
                <SkipForward size={18}/>
              </button>
              <button onClick={fullReset}
                style={{padding:8,background:'rgba(127,29,29,0.5)',color:'#fecaca',border:'1px solid #7f1d1d',borderRadius:8,cursor:'pointer'}} title="重置">
                <RotateCcw size={18}/>
              </button>
            </div>
            <div style={{display:'flex',alignItems:'center',gap:12,background:'#1e293b',padding:8,borderRadius:4,fontSize:12}}>
              <span style={{whiteSpace:'nowrap'}}>速度:</span>
              <input type="range" min="10" max="1000" step="10"
                value={1010 - speed}
                onChange={(e) => setSpeed(1010 - Number(e.target.value))}
                style={{width:'100%',height:4,cursor:'pointer'}}
              />
            </div>
            <div style={{display:'flex',gap:8}}>
              <button onClick={() => { setMode('TRAIN'); setEpsilon(0.5); }}
                style={{flex:1,padding:'4px 0',fontSize:12,borderRadius:4,cursor:'pointer',
                  background: mode === 'TRAIN' ? '#2563eb' : 'transparent',
                  border: mode === 'TRAIN' ? '1px solid #3b82f6' : '1px solid #475569',
                  color: mode === 'TRAIN' ? '#fff' : '#64748b'}}>
                训练模式 (Learn)
              </button>
              <button onClick={() => { setMode('EVAL'); setEpsilon(0); }}
                style={{flex:1,padding:'4px 0',fontSize:12,borderRadius:4,cursor:'pointer',
                  background: mode === 'EVAL' ? '#7c3aed' : 'transparent',
                  border: mode === 'EVAL' ? '1px solid #8b5cf6' : '1px solid #475569',
                  color: mode === 'EVAL' ? '#fff' : '#64748b'}}>
                评估模式 (Eval)
              </button>
            </div>
          </div>
        </div>

        {/* Middle: Neural Network */}
        <div style={{display:'flex',flexDirection:'column',gap:16}}>
          <h3 style={{fontSize:14,fontWeight:'bold',color:'#94a3b8',textTransform:'uppercase',letterSpacing:'0.05em',display:'flex',alignItems:'center',gap:8,margin:0}}>
            <Brain size={16} /> 神经网络 (Q-Network)
          </h3>
          <NetworkViz />
          {/* Explanation */}
          <div style={{flex:1,background:'#0f172a',padding:16,borderRadius:12,border:'1px solid #1e293b',fontSize:14,lineHeight:1.6,color:'#cbd5e1'}}>
            <h4 style={{fontWeight:'bold',color:'#fff',marginBottom:8,borderBottom:'1px solid #334155',paddingBottom:4,marginTop:0}}>当前决策解析</h4>
            {lastAction ? (
              <div style={{display:'flex',flexDirection:'column',gap:8}}>
                <div style={{display:'flex',justifyContent:'space-between'}}>
                  <span>状态 (Input):</span>
                  <span style={{fontFamily:'monospace',color:'#60a5fa'}}>({agentPos.x}, {agentPos.y})</span>
                </div>
                <div style={{display:'flex',justifyContent:'space-between'}}>
                  <span>策略 (Policy):</span>
                  <span style={{color: lastAction.isRandom ? '#fbbf24' : '#c084fc'}}>
                    {lastAction.isRandom ? 'Epsilon-Greedy (随机探索)' : 'Argmax Q (贪婪利用)'}
                  </span>
                </div>
                <div style={{display:'flex',justifyContent:'space-between'}}>
                  <span>动作 (Output):</span>
                  <span style={{fontWeight:'bold',color:'#fff'}}>{lastAction.action}</span>
                </div>
                {mode === 'TRAIN' && (
                  <div style={{marginTop:8,padding:8,background:'#1e293b',borderRadius:4,borderLeft:'2px solid #eab308',fontSize:12}}>
                    <strong>Loss Update:</strong>
                    <div style={{marginTop:4,opacity:0.75}}>
                      Q<sub>new</sub> = Q + α(R + γQ<sub>max</sub> - Q)
                    </div>
                    <div style={{marginTop:4,color:'#94a3b8'}}>
                      网络正在通过TD误差修正刚才的预测。
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div style={{color:'#64748b',fontStyle:'italic',padding:'16px 0',textAlign:'center'}}>
                等待开始... 点击训练按钮。
              </div>
            )}
          </div>
        </div>

        {/* Right: Memory & Loss */}
        <div style={{display:'flex',flexDirection:'column',gap:16}}>
          <h3 style={{fontSize:14,fontWeight:'bold',color:'#94a3b8',textTransform:'uppercase',letterSpacing:'0.05em',display:'flex',alignItems:'center',gap:8,margin:0}}>
            <Database size={16} /> 经验回放 (Replay Buffer)
          </h3>
          {/* Replay Buffer */}
          <div style={{background:'#0f172a',height:256,overflow:'hidden',position:'relative',borderRadius:12,border:'1px solid #1e293b',display:'flex',flexDirection:'column'}}>
            <div style={{position:'absolute',top:0,width:'100%',background:'#1e293b',zIndex:10,padding:8,fontSize:12,textAlign:'center',borderBottom:'1px solid #334155',color:'#94a3b8',boxSizing:'border-box'}}>
              存储最近的经验元组 (s, a, r, s')
            </div>
            <div style={{overflowY:'auto',flex:1,padding:8,paddingTop:40,display:'flex',flexDirection:'column',gap:4}}>
              {memory.length === 0 && <div style={{textAlign:'center',color:'#475569',fontSize:12,marginTop:40}}>Buffer Empty</div>}
              {memory.map((exp) => (
                <div key={exp.id} style={{display:'grid',gridTemplateColumns:'1fr 1fr 1fr 1fr',gap:4,fontSize:10,background:'#1e293b',padding:6,borderRadius:4,borderLeft:'2px solid #3b82f6'}}>
                  <div style={{color:'#94a3b8'}}>S:({exp.state.x},{exp.state.y})</div>
                  <div style={{color:'#eab308',fontWeight:'bold'}}>{exp.action}</div>
                  <div style={{color: exp.reward > 0 ? '#4ade80' : '#f87171'}}>R:{exp.reward}</div>
                  <div style={{color:'#94a3b8'}}>S':({exp.nextState.x},{exp.nextState.y})</div>
                </div>
              ))}
            </div>
          </div>

          <h3 style={{fontSize:14,fontWeight:'bold',color:'#94a3b8',textTransform:'uppercase',letterSpacing:'0.05em',display:'flex',alignItems:'center',gap:8,margin:0,marginTop:8}}>
            <Activity size={16} /> 训练损失 (Loss)
          </h3>
          {/* Loss Chart */}
          <div style={{height:160,background:'#0f172a',borderRadius:12,border:'1px solid #1e293b',position:'relative',padding:8,display:'flex',alignItems:'flex-end'}}>
            {lossHistory.length > 2 ? (
              <svg style={{width:'100%',height:'100%'}} viewBox={`0 0 ${lossHistory.length} 100`} preserveAspectRatio="none">
                <path
                  d={`M 0 100 ` + lossHistory.map((val, i) => {
                    const y = Math.max(0, 100 - (val * 50));
                    return `L ${i} ${y}`;
                  }).join(' ')}
                  fill="none"
                  stroke="#f43f5e"
                  strokeWidth="2"
                />
              </svg>
            ) : (
              <div style={{width:'100%',textAlign:'center',color:'#475569',fontSize:12,alignSelf:'center'}}>暂无 Loss 数据</div>
            )}
            <div style={{position:'absolute',top:8,left:8,fontSize:10,color:'#64748b'}}>Loss (MSE)</div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer style={{marginTop:32,fontSize:12,color:'#64748b',maxWidth:800,textAlign:'center',lineHeight:1.6}}>
        <p>
          <strong>原理说明：</strong> DQN 使用神经网络（此处简化为 Q-Table 可视化）来近似 Q 函数。
          它通过 <strong>Experience Replay（经验回放）</strong> 打破数据相关性，并利用 Target Network（目标网络）思想稳定训练。
          观察上面的"热力图"：随着训练进行，到达终点的路径颜色会变深（Q值变高），陷阱周围会变暗（Q值变低）。
        </p>
      </footer>
    </div>
  );
};

export default DqnVisualization;
