'use client';

import { useState } from 'react';
import {
  ArrowDown,
  ArrowUpRight,
  BriefcaseBusiness,
  Camera,
  ChevronDown,
  Code2,
  Headphones,
  Languages,
  Mail,
  Music2,
  Play,
  X,
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { BackgroundSignal } from '@/components/background-signal';
import { type Language, type ProjectMedia, projects } from '@/lib/projects';

const copy = {
  zh: {
    status: '声音设计师 · 音频程序员 · 声音艺术家',
    welcome: '欢迎来到我的声音世界',
    explore: '进入作品档案',
    selected: '项目档案',
    selectedNote: '8 个项目 · 声音、代码、空间与交互',
    open: '展开项目',
    close: '收起项目',
    overview: '项目简介',
    approach: '创作与技术',
    tools: '创作工具',
    documentation: '图像与结构资料',
    media: '音视频资料',
    noMedia: '该项目未包含可展示的音视频附件。',
    github: '进入 GitHub 项目',
    githubNote: '查看源代码、补丁与完整项目文件',
    playbackNote: '所有音视频均可在页面内直接播放。',
    footer: '持续创造中...',
    switchLabel: 'Switch to English',
    contact: '联系与个人主页',
    email: '邮箱',
  },
  en: {
    status: 'Sound Designer · Audio Programmer · Sound Artist',
    welcome: 'Welcome to my world of sound',
    explore: 'Enter the archive',
    selected: 'Project archive',
    selectedNote: '8 projects · sound, code, space, and interaction',
    open: 'Open project',
    close: 'Close project',
    overview: 'Overview',
    approach: 'Creative & technical approach',
    tools: 'Tools',
    documentation: 'Images & system documents',
    media: 'Audio & video',
    noMedia: 'No audio or video attachment is included in this project.',
    github: 'View GitHub project',
    githubNote: 'Source code, patches, and complete project files',
    playbackNote:
      'Every audio and video work can be played directly on this page.',
    footer: 'Continuously creating...',
    switchLabel: '切换至中文',
    contact: 'Contact & profiles',
    email: 'Email',
  },
} as const;

const backgroundAudio = projects.find((project) => project.slug === 'universal-drone')!
  .media.find((media) => media.kind === 'audio')!;

const groupMedia = (media: ProjectMedia[]) => {
  const groups = new Map<
    string,
    { label: ProjectMedia['group']; items: ProjectMedia[] }
  >();

  media.forEach((item) => {
    const key = item.group.en;
    const group = groups.get(key);
    if (group) group.items.push(item);
    else groups.set(key, { label: item.group, items: [item] });
  });

  return Array.from(groups.values());
};

type MediaGroupProps = {
  group: ReturnType<typeof groupMedia>[number];
  language: Language;
  defaultOpen: boolean;
};

function MediaGroup({ group, language, defaultOpen }: MediaGroupProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <details
      className="media-group"
      open={isOpen}
      onToggle={(event) => setIsOpen(event.currentTarget.open)}
    >
      <summary>
        <span>{group.label[language]}</span>
        <span>{String(group.items.length).padStart(2, '0')}</span>
      </summary>
      {isOpen && (
        <div className="inline-media-list">
          {group.items.map((media) => (
            <article className={`inline-media ${media.kind}`} key={media.src}>
              <div className="inline-media-heading">
                <span className="media-kind" aria-hidden="true">
                  {media.kind === 'audio' ? <Headphones /> : <Play />}
                </span>
                <span>{media.label[language]}</span>
              </div>
              {media.kind === 'audio' ? (
                <audio
                  ref={(audio) => { if (audio) audio.disableRemotePlayback = true; }}
                  controls
                  controlsList="nodownload noplaybackrate noremoteplayback"
                  preload="none"
                  src={media.src}
                  onContextMenu={(event) => event.preventDefault()}
                >
                  {media.label[language]}
                </audio>
              ) : (
                <video
                  controls
                  controlsList="nodownload noplaybackrate noremoteplayback"
                  disablePictureInPicture
                  disableRemotePlayback
                  preload="none"
                  playsInline
                  src={media.src}
                  onContextMenu={(event) => event.preventDefault()}
                >
                  {media.label[language]}
                </video>
              )}
            </article>
          ))}
        </div>
      )}
    </details>
  );
}

export default function Home() {
  const [language, setLanguage] = useState<Language>('zh');
  const [openProject, setOpenProject] = useState<string | null>(null);
  const text = copy[language];

  return (
    <main
      lang={language === 'zh' ? 'zh-CN' : 'en'}
      className="min-h-screen overflow-hidden"
    >
      <header className="site-header">
        <a className="wordmark" href="#top" aria-label="Crescent Feng, home">
          CF<span className="wordmark-dot">●</span>
        </a>

        <p className="archive-code">SOUND / CODE / ART</p>

        <Button
          variant="outline"
          size="sm"
          className="language-button"
          onClick={() => setLanguage(language === 'zh' ? 'en' : 'zh')}
          aria-label={text.switchLabel}
        >
          <Languages aria-hidden="true" />
          {language === 'zh' ? 'EN' : '中文'}
        </Button>
      </header>

      <section id="top" className="hero-section">
        <nav className="hero-index" aria-label={text.contact}>
          <a href="/shengke/index.html" title={language === 'zh' ? '声刻 · 语音克隆与情绪 TTS' : 'Shengke · Voice cloning & emotional TTS'}>
            <Headphones aria-hidden="true" />
            <span>{language === 'zh' ? '声刻 · 开发中' : 'Shengke · In development'}</span>
          </a>
          <a href="mailto:xinyuefeng0131@163.com" title="xinyuefeng0131@163.com" aria-label={`${text.email}: xinyuefeng0131@163.com`}>
            <Mail aria-hidden="true" />
            <span>{text.email}</span>
          </a>
          <a href="https://github.com/CrescentFeng" target="_blank" rel="noopener noreferrer" title="GitHub · CrescentFeng">
            <Code2 aria-hidden="true" />
            <span>GitHub</span>
          </a>
          <a href="https://www.linkedin.com/in/%E6%96%B0%E6%9C%88-%E5%86%AF-848181438/" target="_blank" rel="noopener noreferrer" title="LinkedIn">
            <BriefcaseBusiness aria-hidden="true" />
            <span>LinkedIn</span>
          </a>
          <a href="https://www.instagram.com/crescentfff/" target="_blank" rel="noopener noreferrer" title="Instagram · @crescentfff">
            <Camera aria-hidden="true" />
            <span>Instagram</span>
          </a>
        </nav>

        <div className="hero-main">
          <div className="hero-intro-block">
            <p className="eyebrow">{text.status}</p>
            <p className="hero-welcome">
              {text.welcome}
              <Music2
                aria-hidden="true"
                style={{ display: 'inline-block', width: '1em', height: '1em', marginLeft: '0.35em', verticalAlign: '-0.1em' }}
              />
            </p>
          </div>
          <h1 className="hero-name">
            <span className="name-primary">
              {language === 'zh' ? '冯新月' : 'CRESCENT'}
            </span>
            <span className="name-secondary">
              {language === 'zh' ? 'XINYUE FENG' : 'FENG'}
            </span>
          </h1>
          <div className="hero-lower">
            <span />
            <a className="archive-link" href="#archive">
              {text.explore}
              <ArrowDown aria-hidden="true" />
            </a>
          </div>
        </div>

        <BackgroundSignal language={language} src={backgroundAudio.src} />
      </section>

      <section id="archive" className="archive-section">
        <div className="section-heading">
          <div>
            <p>01 — 08</p>
            <h2>{text.selected}</h2>
          </div>
          <p>{text.selectedNote}</p>
        </div>

        <div className="project-list">
          {projects.map((project) => {
            const isOpen = openProject === project.slug;
            const mediaGroups = groupMedia(project.media);

            return (
              <article
                className={`project-entry ${isOpen ? 'is-open' : ''}`}
                key={project.slug}
              >
                <button
                  className="project-trigger"
                  onClick={() => setOpenProject(isOpen ? null : project.slug)}
                  aria-expanded={isOpen}
                  aria-controls={`project-${project.slug}`}
                >
                  <span className="project-number">{project.number}</span>
                  <span className="project-title-block">
                    <span className="project-kicker">
                      {project.category[language]}
                    </span>
                    <span className="project-title">{project.title}</span>
                  </span>
                  <span className="project-tools-preview">
                    {project.tools.slice(0, 3).join(' / ')}
                  </span>
                  <span className="project-open-label">
                    {isOpen ? text.close : text.open}
                  </span>
                  <span className="project-toggle-icon" aria-hidden="true">
                    {isOpen ? <X /> : <ChevronDown />}
                  </span>
                </button>

                {isOpen && (
                  <div
                    className="project-detail"
                    id={`project-${project.slug}`}
                  >
                    <div className="detail-copy">
                      <section>
                        <p className="detail-label">{text.overview}</p>
                        <p className="detail-lead">
                          {project.summary[language]}
                        </p>
                      </section>
                      <section>
                        <p className="detail-label">{text.approach}</p>
                        <p>{project.approach[language]}</p>
                      </section>
                      <section>
                        <p className="detail-label">{text.tools}</p>
                        <ul className="tool-list" aria-label={text.tools}>
                          {project.tools.map((tool) => (
                            <li key={tool}>{tool}</li>
                          ))}
                        </ul>
                      </section>
                    </div>

                    <div className="detail-assets">
                      {project.images.length > 0 && (
                        <details className="asset-section documentation-group">
                          <summary className="asset-heading">
                            <p className="detail-label">{text.documentation}</p>
                            <span>
                              {String(project.images.length).padStart(2, '0')}
                            </span>
                          </summary>
                          <div
                            className={`image-gallery ${project.images.length === 1 ? 'single' : ''}`}
                          >
                            {project.images.map((image) => (
                              <figure key={image.src}>
                                <img
                                  src={image.src}
                                  alt={image.alt[language]}
                                  loading="lazy"
                                  draggable={false}
                                  onContextMenu={(event) =>
                                    event.preventDefault()
                                  }
                                  onDragStart={(event) =>
                                    event.preventDefault()
                                  }
                                />
                                <figcaption>
                                  {image.caption[language]}
                                </figcaption>
                              </figure>
                            ))}
                          </div>
                        </details>
                      )}

                      <details className="asset-section documentation-group">
                        <summary className="asset-heading">
                          <p className="detail-label">{text.media}</p>
                          <span>
                            {String(project.media.length).padStart(2, '0')}
                          </span>
                        </summary>
                        {project.media.length > 0 ? (
                          <div className="media-groups">
                            {mediaGroups.map((group) => (
                              <MediaGroup
                                key={group.label.en}
                                group={group}
                                language={language}
                                defaultOpen={mediaGroups.length === 1}
                              />
                            ))}
                          </div>
                        ) : (
                          <p className="no-media">{text.noMedia}</p>
                        )}
                        {project.media.length > 0 && (
                          <p className="playback-note">{text.playbackNote}</p>
                        )}
                      </details>

                      <a
                        className="github-link"
                        href={project.github}
                        target="_blank"
                        rel="noreferrer"
                      >
                        <Code2 aria-hidden="true" />
                        <span>
                          <strong>{text.github}</strong>
                          <small>{text.githubNote}</small>
                        </span>
                        <ArrowUpRight aria-hidden="true" />
                      </a>
                    </div>
                  </div>
                )}
              </article>
            );
          })}
        </div>
      </section>

      <footer className="site-footer">
        <p>{text.footer}</p>
        <p>CRESCENTFFF.COM — SOON</p>
      </footer>
    </main>
  );
}
