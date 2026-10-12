'use strict';
const $ = id => document.getElementById(id);
const app = { arm: 'R_BCE', files: [], jobId: null, job: null, results: [], selected: 0, polling: null, uploading: false }; 

function toast(msg) { const n = $('toast'); n.textContent = String(msg); n.hidden = false; clearTimeout(toast.timer); toast.timer = setTimeout(() => {n.hidden = true;}, 5000); }
async function api(path, options) {
  const response = await fetch(path, options);
  const type = response.headers.get('content-type') || '';
  const value = type.includes('application/json') ? await response.json() : await response.text();
  if (!response.ok) throw new Error(value?.error || '请求失败: ' + response.status);
  return value;
}
function setTab(name) {
  document.querySelectorAll('.nav-item').forEach(el => el.classList.toggle('active', el.dataset.tab === name));
  document.querySelectorAll('.screen').forEach(el => el.classList.toggle('active', el.id === 'tab-' + name));
  if (name === 'history') refreshHistory();
}
function setArm(arm) {
  app.arm = arm;
  document.querySelectorAll('[data-model]').forEach(el => el.classList.toggle('selected', el.dataset.model === arm));
}
function showFiles(files) {
  const unique = [];
  const signatures = new Set();
  for (const file of files) {
    const suffix = '.' + (file.name.split('.').pop() || '').toLowerCase();
    if (!['.jpg','.jpeg','.png','.webp'].includes(suffix)) continue;
    if (file.size > 32 * 1024 * 1024) {toast('超过 32 MiB 的图片已跳过: ' + file.name); continue;}
    const sig = file.name + '/' + file.size + '/' + file.lastModified;
    if (!signatures.has(sig)) { signatures.add(sig); unique.push(file); }
    if (unique.length >= 200) break;
  }
  app.files = unique;
  $('file-count').textContent = unique.length ? `已选择 ${unique.length} 张图片` : '当前未选择图片';
  $('file-list').replaceChildren();
  unique.slice(0,7).forEach(f => {const d = document.createElement('div');d.textContent = '▹  '+f.name; $('file-list').append(d);});
  if (unique.length > 7) {const d = document.createElement('div');d.textContent = `另有 ${unique.length - 7} 张…`; $('file-list').append(d);}
  $('start-button').disabled = unique.length === 0;
}
function progress(text, done, total) {
  $('progress-wrap').hidden = false;
  $('progress-label').textContent = text;
  $('progress-number').textContent = `${done}/${total}`;
  $('progress-fill').style.width = `${total ? 100*done/total : 0}%`;
}
function busy(working) { $('start-button').disabled = working || !app.files.length; document.querySelectorAll('[data-model]').forEach(x=>x.disabled=working); }
async function startJob() {
  if (!app.files.length) return;
  busy(true); app.uploading = true;
  try {
    const ids=[];
    for (let i=0;i<app.files.length;i++) {
      const file=app.files[i];
      progress('正在导入本地图片',i,app.files.length);
      const item=await api('/api/uploads', { method:'POST', headers:{'Content-Type':'application/octet-stream', 'X-File-Name':encodeURIComponent(file.name)}, body: file });
      ids.push(item.id);
    }
    progress('等待 NPU 处理',0,ids.length);
    const j=await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({arm:app.arm,image_ids:ids})});
    app.uploading = false;
    app.jobId=j.id;
    window.localStorage.setItem('eviroute_latest_job',j.id);
    setTab('detect');
    await poll();
  } catch(err) { app.uploading = false; toast('提交失败：' + err.message); busy(false); }
}
async function poll() {
  if (app.polling) clearTimeout(app.polling);
  if (!app.jobId) return;
  try {
    const job=await api('/api/jobs/'+app.jobId);
    app.job=job;
    progress(job.status==='running'?'NPU 本地检测中':job.status==='queued'?'排队等待':'任务已结束', job.done, job.total);
    if (job.status==='completed') {
      progress('全部完成',job.total,job.total);
      const result=await api('/api/jobs/'+job.id+'/results');
      app.results=result.results;
      app.selected=0;
      renderResults();
      $('export-button').disabled=false;
      busy(false);
      return;
    }
    if (job.status==='failed' || job.status==='interrupted') {
      toast('任务未完成：' + (job.error || job.status));
      busy(false);
      return;
    }
    app.polling=setTimeout(poll,700);
  } catch(err) { toast('读取任务失败：' + err.message); busy(false); }
}
function renderResults() {
  const job=app.job;
  $('detect-empty').hidden=true;
  $('detect-results').hidden=false;
  $('stat-model').textContent=job.arm==='R_BCE'?'R-BCE':'R-Debias';
  $('stat-finished').textContent=`${job.total} / ${job.total}`;
  $('stat-fps').textContent=(job.summary?.processing_fps?.toFixed(2) || '—') + ' FPS';
  $('list-count').textContent=`(${app.results.length})`;
  const list=$('result-list'); list.replaceChildren();
  app.results.forEach((r,i)=>{
    const button=document.createElement('button');
    button.type='button'; button.className='result-item';
    const name=document.createElement('span');name.className='file';name.textContent=(i+1)+'. '+r.filename;
    const outcome=document.createElement('span');outcome.className='tag '+r.prediction.toLowerCase();outcome.textContent=r.prediction;
    button.append(name,outcome);
    button.onclick=()=>selectResult(i);
    list.append(button);
  });
  selectResult(0);
}
function selectResult(i) {
  const r=app.results[i];if(!r)return;
  app.selected=i;
  document.querySelectorAll('.result-item').forEach((n,k)=>n.classList.toggle('active',k===i));
  $('preview-image').src=r.visual_url || r.image_url;
  $('preview-title').textContent='Top-10 证据可视化';
  $('image-name').textContent=r.filename;
  $('image-pos').textContent=`${i+1} / ${app.results.length}`;
  $('stat-result').textContent=r.prediction;
  $('decision-name').textContent=r.prediction==='FAKE'?'疑似生成':'倾向真实';
  $('decision-name').className='decision-name '+r.prediction.toLowerCase();
  $('decision-desc').textContent=r.prediction==='FAKE'?'Fake-support Logit ≥ 0':'Fake-support Logit < 0';
  $('metric-logit').textContent=(r.logit>=0?'+':'')+r.logit.toFixed(4);
  $('metric-score').textContent=(r.support_score*100).toFixed(2)+'%';
  $('metric-b1').textContent=r.top10_region_counts.b1_count + ' / 10';
}
async function refreshHistory() {
  try {
    const d=await api('/api/jobs'); const list=$('history-list');list.replaceChildren();
    if(!d.jobs.length) {list.textContent='暂无历史任务。';return;}
    for(const j of d.jobs){
      const el=document.createElement('button');el.className='history-item';el.type='button';
      const meta=document.createElement('div');
      const title=document.createElement('strong');title.textContent=(j.arm==='R_BCE'?'R-BCE':'R-Debias')+' · '+j.total+' 张图片';
      const detail=document.createElement('small');detail.textContent=new Date(j.created_at).toLocaleString('zh-CN')+' · '+j.id.slice(0,8);
      meta.append(title,detail);
      const status=document.createElement('span');status.className='history-status';status.textContent=({completed:'已完成',running:'运行中',queued:'排队中',failed:'失败',interrupted:'中断'})[j.status] || j.status;
      el.append(meta,status);
      el.onclick=()=>{app.jobId=j.id; app.job=j; setTab('detect'); if(j.status==='completed'){poll();}else{poll();}};
      list.append(el);
    }
  }catch(err){toast('历史记录读取失败：'+err.message);}
}
async function checkStatus(){
  try { const data=await api('/api/status');const b=data.arms.R_BCE,d=data.arms.R_Debias;
    $('device-status').textContent=(b&&d?'● 双模型就绪':'● 请检查模型') + ' / NPU Core 0_1_2';
    $('device-status').style.color=b&&d?'#4ad6bb':'#f1bb80';
  }catch(_){$('device-status').textContent='● 本地服务暂不可用';$('device-status').style.color='#f18f84';}
}
async function followExternalJob() {
  // SSH-submitted jobs are shown automatically on the HDMI display;
  // no mouse or touchscreen is required for the first prototype test.
  if (app.uploading || (app.job && ['queued','running'].includes(app.job.status))) return;
  try {
    const feed=await api('/api/jobs');
    const latest=feed.jobs[0];
    if (!latest || latest.id===app.jobId) return;
    app.jobId=latest.id;
    window.localStorage.setItem('eviroute_latest_job', latest.id);
    setTab('detect');
    await poll();
  } catch(_){ /* offline service can be temporarily restarting */ }
}
function setup(){
  document.querySelectorAll('.nav-item').forEach(el=>el.addEventListener('click',()=>setTab(el.dataset.tab)));
  document.querySelectorAll('[data-model]').forEach(el=>el.addEventListener('click',()=>setArm(el.dataset.model)));
  $('file-input').addEventListener('change',e=>showFiles([...e.target.files]));
  const dz=$('dropzone');
  dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('drag');});
  dz.addEventListener('dragleave',()=>dz.classList.remove('drag'));
  dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('drag');showFiles([...e.dataTransfer.files]);});
  $('start-button').addEventListener('click',startJob);
  $('export-button').addEventListener('click',()=>{if(app.jobId)location.href='/api/jobs/'+app.jobId+'/export';});
  $('refresh-history').addEventListener('click',refreshHistory);
  function clock(){ $('clock').textContent=new Date().toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'}); }
  clock();setInterval(clock,60000);checkStatus();
  const previous=window.localStorage.getItem('eviroute_latest_job');
  if(previous && /^[0-9a-f]{32}$/.test(previous)){app.jobId=previous;poll();}
  else { followExternalJob(); }
  setInterval(followExternalJob, 3000);
}
window.addEventListener('DOMContentLoaded',setup);
