
# Run from ClinicalSignBridge repository root

$SourceRoot = Join-Path (Get-Location) "ai-engine"
$ArchiveFolder = Join-Path $SourceRoot "archive\experimental-scripts-2026-11-16"

# Safety switch: change to $true only after reviewing the preview.
$Apply = $true

$Files = @(
    # Hand association experiments
    "associate_hand_slots.py",
    "associate_hand_slots_v2.py",
    "evaluate_association_v2.py",
    "validate_association_v2.py",

    # Hand detection and landmark diagnostics
    "analyze_association_gaps.py",
    "analyze_hand_detection_gaps.py",
    "analyze_hand_detection_patterns.py",
    "analyze_hand_slot_consistency.py",
    "analyze_hand_slot_swaps.py",
    "analyze_handedness_errors.py",
    "analyze_landmark_quality.py",
    "analyze_standalone_temporal_quality.py",
    "analyze_v2_rejections.py",

    # Inspection and debugging
    "compare_original_and_associated.py",
    "diagnose_frame4_association.py",
    "extract_one_video.py",
    "extract_standalone_hands.py",
    "inspect_download_requirements.py",
    "inspect_duplicate_paths.py",
    "inspect_extracted_features.py",
    "inspect_handedness_frames.py",
    "inspect_include.py",
    "inspect_problem_videos.py",
    "inspect_slot_swap.py",
    "inspect_zenodo_archives.py",
    "profile_include_dataset.py",
    "verify_video.py",

    # Old training experiments
    "train_baseline.py",
    "export_bilstm_test_errors.py",

    # Standalone diagnostic probes
    "test_detector_frame_comparison.py",
    "test_hand_detection_modes.py",
    "test_hand_detection_resolution.py",
    "test_mediapipe.py",
    "test_single_frame_gap.py",
    "test_single_video_download.py",
    "test_standalone_hand_detection.py"
)

if (-not (Test-Path $SourceRoot)) {
    throw "Source folder not found: $SourceRoot. Run this script from the repository root."
}

$ExistingFiles = @(
    $Files | Where-Object {
        Test-Path (Join-Path $SourceRoot $_)
    }
)

$MissingFiles = @(
    $Files | Where-Object {
        -not (Test-Path (Join-Path $SourceRoot $_))
    }
)

Write-Host "`nExperimental scripts found: $($ExistingFiles.Count)" -ForegroundColor Cyan
Write-Host "Scripts not found: $($MissingFiles.Count)" -ForegroundColor Yellow

if ($MissingFiles.Count -gt 0) {
    Write-Host "`nNot found (left untouched):"
    $MissingFiles | ForEach-Object { Write-Host "  $_" }
}

if ($ExistingFiles.Count -eq 0) {
    Write-Host "`nNothing to archive."
    return
}

if (-not $Apply) {
    Write-Host "`nPREVIEW ONLY. The following files would be archived:" -ForegroundColor Yellow
    $ExistingFiles | ForEach-Object { Write-Host "  $_" }
    Write-Host "`nNo files were moved."
    Write-Host "Review the list, then set `$Apply = `$true and run again."
    return
}

if (Test-Path $ArchiveFolder) {
    throw "Archive folder already exists. Choose a new archive name to avoid overwriting files: $ArchiveFolder"
}

New-Item -ItemType Directory -Path $ArchiveFolder -Force | Out-Null

foreach ($File in $ExistingFiles) {
    $Source = Join-Path $SourceRoot $File
    $Destination = Join-Path $ArchiveFolder $File

    Move-Item -LiteralPath $Source -Destination $Destination
    Write-Host "Archived: $File" -ForegroundColor Green
}

Write-Host "`nArchive completed." -ForegroundColor Green
Write-Host "Location: $ArchiveFolder"
Write-Host "Files moved: $($ExistingFiles.Count)"
