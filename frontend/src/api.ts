// Type definitions matching backend schemas
export interface ImageStats {
  width: number
  height: number
  channels: number
  total_pixels: number
  file_size_mb: number
  mean_brightness: number
  std_brightness: number
}

export interface WeedDetection {
  areas_detected: number
  total_weed_area_pixels: number
  weed_coverage_percentage: number
  detection_sensitivity: number
}

export interface ProcessingSummary {
  processing_time_seconds: number
  image_stats: ImageStats
  weed_detection: WeedDetection
  analysis_status: string
  detected_issues: string[]
  scale_factor?: number
}

export interface ProcessingParameters {
  sensitivity: number
  algorithm: string
  version: string
}

export interface ProcessResult {
  success: boolean
  result_image_url: string
  summary: ProcessingSummary
  weed_polygons: number[][][]
  analysis_notes: string
  processing_parameters: ProcessingParameters
}

export interface ProcessStatusResponse {
  status: string
  algorithms_available: string[]
  supported_formats: string[]
  max_file_size_mb: number
  max_image_dimension: number
  version: string
}

export async function uploadImage(file: File, sensitivity: number = 0.5, algorithm: string = 'oblique_pipeline'): Promise<ProcessResult> {
  const fd = new FormData()
  fd.append('file', file)
  
  const url = `/api/process?sensitivity=${sensitivity}&algorithm=${algorithm}`
  const res = await fetch(url, { method: 'POST', body: fd })
  
  if (!res.ok) {
    let errorMessage = 'Falha ao processar imagem'
    try {
      const errorData = await res.json()
      errorMessage = errorData.detail || errorMessage
    } catch {
      const txt = await res.text().catch(() => '')
      if (txt) errorMessage += `: ${txt}`
    }
    throw new Error(errorMessage)
  }
  
  return res.json()
}

export async function getProcessingStatus(): Promise<ProcessStatusResponse> {
  const res = await fetch('/api/process/status')
  
  if (!res.ok) {
    throw new Error('Falha ao obter status do sistema')
  }
  
  return res.json()
}

// Legacy function for backward compatibility
export async function uploadImageLegacy(file: File): Promise<{result_image_url: string, notes?: string}> {
  const result = await uploadImage(file, 0.5)
  return {
    result_image_url: result.result_image_url,
    notes: result.analysis_notes
  }
}
