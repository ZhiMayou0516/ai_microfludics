const { BASE_URL } = require('./config')

function uploadAnalyze({ filePath, fileName, sampleId, mode, dropFirstRows, smoothWindow, dropTimeColumn }) {
  return new Promise((resolve, reject) => {
    wx.uploadFile({
      url: `${BASE_URL}/api/analyze`,
      filePath,
      name: 'file',
      fileName: fileName || 'curve.csv',
      formData: {
        sample_id: sampleId || '',
        mode: mode || 'rule',
        drop_first_rows: String(dropFirstRows ?? 5),
        smooth_window: String(smoothWindow ?? 5),
        drop_time_column: dropTimeColumn ? 'true' : 'false',
        original_filename: fileName || 'curve.csv'
      },
      success(res) {
        let data = null
        try {
          data = JSON.parse(res.data)
        } catch (e) {
          reject(new Error('后端返回内容不是 JSON：' + res.data))
          return
        }
        if (res.statusCode >= 200 && res.statusCode < 300 && data.ok) {
          resolve(data)
        } else {
          const msg = data.detail || data.message || '分析失败'
          reject(new Error(typeof msg === 'string' ? msg : JSON.stringify(msg)))
        }
      },
      fail(err) {
        reject(new Error(err.errMsg || '无法连接后端'))
      }
    })
  })
}

module.exports = {
  uploadAnalyze
}
